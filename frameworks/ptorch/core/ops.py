import torch
import torch.nn.functional as F
import warnings
from ..config import config
import math
from itertools import repeat


# Precomputed optimal coefficients from Polar Express (degree=5)
# Paper: "The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm"
_POLAR_COEFFS = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375)
]

# Apply safety factor for numerical stability in bfloat16 (Section 3.4 & Appendix G)
_POLAR_COEFFS = [
    (a / 1.01, b / 1.01**3, c / 1.01**5) for a, b, c in _POLAR_COEFFS[:-1]
] + [_POLAR_COEFFS[-1]]

@torch.compile()
def zeropower_via_polarexpress(G: torch.Tensor, steps: int = 5, eps: float = 1e-2) -> torch.Tensor:
    """
    Computes the polar factor using the optimal Polar Express polynomial method.
    Acts as a drop-in replacement for the static Newton-Schulz5 / Jordan method.
    """
    is_1d = G.ndim == 1
    if is_1d:
        G = G.view(1, -1)
        
    X = G.bfloat16()
    
    # As per paper recommendation: use a larger eps (1e-2) to avoid conditioning issues
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    
    # Transpose if rows > cols to save FLOPs in matrix multiplications
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
        
    # Build sequence of coefficients for requested number of steps
    hs = _POLAR_COEFFS[:steps]
    if steps > len(_POLAR_COEFFS):
        hs += list(repeat(_POLAR_COEFFS[-1], steps - len(_POLAR_COEFFS)))
        
    # Iterate dynamically shifting polynomials
    for a, b, c in hs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
        
    out = X.mT if transposed else X
    
    if is_1d:
        out = out.view(-1)
        
    return out.to(G.dtype)

@torch.compile()
def process_activation_target(A_det, A_proj):
    if not config.muon_activations:
        return A_proj

    orig_shape = A_det.shape
    last_dim = orig_shape[-1]

    if last_dim == 0 or A_det.numel() == 0:
        return A_proj

    A_2d = A_det.reshape(-1, last_dim)
    A_proj_2d = A_proj.reshape(-1, last_dim)
    g = A_2d - A_proj_2d

    g_muon = zeropower_via_polarexpress(g)

    if config.muon_activations_scale:
        scale = g.norm() / (g_muon.norm() + 1e-8)
        g_muon = g_muon * scale

    A_proj_new = A_2d - config.muon_activations_lr * g_muon

    if config.muon_activations_norm_preserve:
        row_norm_orig = A_2d.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        row_norm_new = A_proj_new.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        A_proj_new = A_proj_new * (row_norm_orig / row_norm_new)

    return A_proj_new.reshape(orig_shape)


@torch.compile()
def process_weight_target(B_det, B_proj):
    if config.muon_weights:
        if B_det.numel() == 0:
            return B_proj

        is_1d = B_det.ndim == 1
        if is_1d:
            B_det = B_det.unsqueeze(0)
            B_proj = B_proj.unsqueeze(0)

        orig_shape = B_det.shape
        B_2d = B_det.reshape(-1, orig_shape[-1])
        B_proj_2d = B_proj.reshape(-1, orig_shape[-1])
        g = B_2d - B_proj_2d

        g_muon = zeropower_via_polarexpress(g)

        if config.muon_weights_scale:
            scale = g.norm() / (g_muon.norm() + 1e-8)
            g_muon = g_muon * scale

        B_new = B_2d - config.muon_weights_lr * g_muon
        result = B_new.reshape(orig_shape)
        if is_1d:
            result = result.squeeze(0)
        return result
    return B_proj


def compute_weight_target_frozen_a(A, B, Z_target_scaled, g=1.0, omega=1.0, residual=False):
    """Optimal weight target with frozen activations via (K,K) least-squares solve.

    Solves: min ||B_new - B||^2 + (1/g^2)||A@B_new - Z_target_scaled||^2
    where Z_target_scaled = Z_target * omega.
    """
    K = B.size(-2)
    frozen_g = g * config.frozen_a_g
    lam = 1.0 / (frozen_g ** 2 + 1e-8)

    if residual:
        I_B = torch.eye(K, B.size(-1), device=B.device, dtype=B.dtype)
        B_eff = I_B - B
    else:
        B_eff = B

    ATA = A.transpose(-2, -1) @ A
    I_K = torch.eye(K, device=A.device, dtype=A.dtype)
    rhs = B_eff + lam * (A.transpose(-2, -1) @ Z_target_scaled)
    B_eff_new = torch.linalg.solve(I_K + lam * ATA, rhs)

    if residual:
        return I_B - B_eff_new
    return B_eff_new


class AverageGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_paths=2):
        ctx.num_paths = num_paths
        
        # .clone() is critical here. It creates a distinct new node in the 
        # computational graph to accumulate the summed gradients from the split.
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output is the sum of the gradients from all paths.
        # We divide it to get the average.
        grad_input = grad_output / ctx.num_paths
        
        # Return a gradient for every input to forward().
        # num_paths is an int, so it requires no gradient (None).
        return grad_input, None
        

@torch.compile()
def matmul_proj_linf(A, B, Z, eps_init=None, g=1.0, omega=1.0, num_steps=5, residual=False):
    """
    Exact independent bilinear projection for A @ B = Z using the L_infinity (Chebyshev) norm.
    
    Analytically optimized using a 45-degree coordinate rotation to transform the L_inf square 
    into an L_1 diamond. This exposes exactly one crossover point per dimension, allowing 
    Newton's method to resolve the piecewise non-linear boundaries in just a few steps.
    
    NOTE: While L_2 is analytically separable, L_inf is not. We rely on `torch.compile` 
    to aggressively fuse the (M, K, N) pointwise expansions into the sum reduction, 
    preventing O(M*N*K) memory materialization in global VRAM.
    """
    M = A.size(-2)
    N = B.size(-1)

    # Apply isomorphism transformation for residual constraint
    if residual:
        I = torch.eye(B.size(-2), B.size(-1), device=B.device, dtype=B.dtype)
        B_eff = I - B
    else:
        B_eff = B

    # 1. Base pairwise projection
    P = A @ B_eff  # (..., M, N)
    Z_target = Z * omega
    
    # Determine the direction of projection for each (m, n) pair
    # S = 1  => Case A (P < Z_target, need to increase P, decrease target)
    # S = -1 => Case B (P > Z_target, need to decrease P, increase target)
    S = torch.sign(Z_target - P) 
    D_abs = torch.abs(Z_target - P)

    # Caching logic for eps (the L_inf radius)
    if eps_init is not None and eps_init.shape == P.shape:
        eps = eps_init.to(P.device)
    else:
        eps = torch.zeros_like(P)

    # 2. Precompute coordinate rotations (M, 1, K) and (1, N, K) 
    # Expanded implicitly to (M, N, K) during fused operations
    A_ext = A.unsqueeze(-1)       # (..., M, K, 1)
    B_ext = B_eff.unsqueeze(-3)   # (..., 1, K, N)
    S_ext = S.unsqueeze(-2)       # (..., M, 1, N)
    
    U_sum = A_ext + B_ext
    V_diff = A_ext - B_ext
    
    u_abs = U_sum.abs() * 0.5
    v_abs = V_diff.abs() * 0.5
    
    S_plus = torch.sign(U_sum)
    S_minus = torch.sign(V_diff)
    
    # Align u and v based on the geometric case (S)
    # S == 1  -> term_A tracks u_abs, term_B tracks v_abs
    # S == -1 -> term_A tracks v_abs, term_B tracks u_abs
    is_case_A = (S_ext == 1)
    
    base_A = torch.where(is_case_A, u_abs, v_abs)
    base_B = torch.where(is_case_A, v_abs, u_abs)
    
    target_penalty = omega / g
    damping = 1e-5
    
    # 3. Newton's Method to find the optimal L_inf boundary (eps)
    for _ in range(num_steps):
        eps_ext = eps.unsqueeze(-2) # (..., M, 1, N)
        
        # Evaluate piecewise branches for the expanding L1 diamond
        term_A = 2.0 * eps_ext * base_A + eps_ext.square()
        term_B = 2.0 * eps_ext * base_B - eps_ext.square()
        
        # Determine which vertices of the diamond have crossed the axis
        wins_A = term_A >= term_B # (..., M, K, N)
        
        # Reconstruct the objective function and derivatives piecewise
        max_terms = torch.where(wins_A, term_A, term_B)
        f_val = max_terms.sum(dim=-2) + eps * target_penalty - D_abs
        
        deriv_A = 2.0 * base_A + 2.0 * eps_ext
        deriv_B = 2.0 * base_B - 2.0 * eps_ext
        
        f_prime_k = torch.where(wins_A, deriv_A, deriv_B)
        f_prime_val = f_prime_k.sum(dim=-2) + target_penalty
        
        # Damped Newton step
        step = f_val / (f_prime_val + damping)
        
        # L_inf radius must remain strictly positive
        eps = torch.clamp(eps - step, min=0.0)

    # 4. Analytical Consensus Reconstruction
    eps_ext = eps.unsqueeze(-2)
    
    # Re-evaluate winning branches with the highly accurate, final eps
    term_A = 2.0 * eps_ext * base_A + eps_ext.square()
    term_B = 2.0 * eps_ext * base_B - eps_ext.square()
    wins_A = term_A >= term_B
    
    # Route the delta updates based on the exact geometric branch taken
    dA_dir = torch.where(
        is_case_A,
        torch.where(wins_A, S_plus, -S_minus),
        torch.where(wins_A, S_minus, -S_plus)
    )
    
    dB_dir = torch.where(
        is_case_A,
        torch.where(wins_A, S_plus, S_minus),
        torch.where(wins_A, -S_minus, -S_plus)
    )
    
    dA = eps_ext * dA_dir
    dB = eps_ext * dB_dir
    
    # Average the independent point-wise proposals
    A_proj = A + dA.mean(dim=-1)
    B_eff_proj = B_eff + dB.mean(dim=-3)
    
    Z_proj = Z - S * (eps / (g * omega))
    
    # Revert isomorphism transformation to get the actual B_proj
    if residual:
        B_proj = I - B_eff_proj
    else:
        B_proj = B_eff_proj
        
    return A_proj, B_proj, Z_proj, eps.detach()


# ─── autograd.Function wrapper ────────────────────────────────────────────────
class MatMulProjectionLinf(torch.autograd.Function):
    """
    Minimize || A_{new} - A_{old} ||_inf + || B_{new} - B_{old} ||_inf + g * || Z_{new} - Z_{old} ||_inf
    subject to A_{new} @ B_{new} = Z_{new} (or A_{new} - A_{new} @ B_{new} = Z_{new} if residual)
    """
    @staticmethod
    def forward(ctx, A, B, num_steps, g, omega, proj_cache=None, pairwise=False, residual=True):
        ctx.save_for_backward(A, B)
        ctx.g = g
        ctx.num_steps = num_steps
        ctx.proj_cache = proj_cache  
        ctx.omega = omega
        ctx.pairwise = pairwise
        ctx.residual = residual
        
        if residual:
            return (A - A @ B) / omega
        else:
            return (A @ B) / omega

    @staticmethod
    def backward(ctx, Z_target):
        A, B = ctx.saved_tensors
        A_det = A.detach()
        B_det = B.detach()
        Z_det = Z_target.detach()

        eps_init = None
        if not ctx.pairwise and ctx.proj_cache is not None:
            eps_init = ctx.proj_cache.get('eps')

        if not ctx.pairwise and (A_det.ndim > 2 or B_det.ndim > 2):
            A_2d = A_det.reshape(-1, A_det.shape[-1]).clone()
            Z_2d = Z_det.reshape(-1, Z_det.shape[-1]).clone() * ctx.omega
            B_2d = B_det.reshape(-1, B_det.shape[-2], B_det.shape[-1]).mean(dim=0).clone()
            
            eps_init_2d = eps_init.reshape(Z_2d.shape) if (eps_init is not None and eps_init.shape == Z_det.shape) else None
            
            A_proj_2d, B_proj_2d, _, eps_new = matmul_proj_linf(
                A_2d, B_2d, Z_2d, eps_init=eps_init_2d, g=ctx.g, 
                omega=ctx.omega, num_steps=ctx.num_steps, residual=ctx.residual
            )
            
            A_proj = A_proj_2d.reshape(A_det.shape)
            B_proj = B_proj_2d.reshape(B_det.shape[-2], B_det.shape[-1]).expand(B_det.shape)
            
            if ctx.proj_cache is not None:
                ctx.proj_cache['eps'] = eps_new.reshape(Z_det.shape)
        else:
            A_proj, B_proj, _, eps_new = matmul_proj_linf(
                A_det.contiguous().clone(), B_det.contiguous().clone(), Z_det.contiguous().clone() * ctx.omega,
                eps_init=eps_init, g=ctx.g, omega=ctx.omega, 
                num_steps=ctx.num_steps, residual=ctx.residual
            )

            if not ctx.pairwise and ctx.proj_cache is not None:
                ctx.proj_cache['eps'] = eps_new

        # Note: the alpha scalar weighting is intentionally removed for L_inf
        return process_activation_target(A_det, A_proj), B_proj, None, None, None, None, None, None        

@torch.compile()
def matmul_proj(A, B, Z, t_init=None, alpha=1.0, g=1.0, omega=1.0, num_steps=1, residual=False):
    """
    Exact independent bilinear projection for A @ B = Z (or A - A @ B = Z if residual=True).
    
    Analytically optimized to avoid O(M*N*K) memory expansions. 
    Pointwise Newton method runs in O(M*N) with numerical safeguards for mixed precision.
    """
    M = A.size(-2)
    N = B.size(-1)

    # Apply isomorphism transformation for residual constraint
    if residual:
        I = torch.eye(B.size(-2), B.size(-1), device=B.device, dtype=B.dtype)
        B_eff = I - B
    else:
        B_eff = B

    # 1. Compute pairwise operations in O(M*N) without 3D expansion
    p = A @ B_eff  # (..., M, N)
    qa = (A * A).sum(dim=-1, keepdim=True)  # (..., M, 1)
    qb = (B_eff * B_eff).sum(dim=-2, keepdim=True)  # (..., 1, N)
    q_eff = qa + alpha * qb  # (..., M, N)

    # 2. Caching logic for t
    if t_init is not None and t_init.shape == p.shape:
        t = t_init.to(p.device)
    else:
        t = torch.zeros_like(p)

    target_penalty = (omega ** 2) / (g ** 2)

    # --- NUMERICAL SAFEGUARDS ---
    eps = 1e-5             # Minimum distance from the singularity
    damping = 1e-4         # Levenberg-Marquardt damping factor
    max_step_size = 0.5    # Maximum allowable change in t per step
    
    # Calculate strict boundary for t to prevent alpha - t^2 from approaching 0
    # We require alpha - t^2 >= eps  =>  t^2 <= alpha - eps
    max_t_val = math.sqrt(max(alpha - eps, eps))
    
    # 3. Newton's Method (fused pointwise ops, numerically stabilized)
    for _ in range(num_steps):
        # Enforce singularity boundary before computing fractions
        t = torch.clamp(t, min=-max_t_val, max=max_t_val)
        
        t2 = t.square()
        alpha_minus_t2 = alpha - t2 # Guaranteed to be >= eps

        N_num = alpha * (p * (alpha + t2) + t * q_eff)
        f_val = (N_num / (alpha_minus_t2.square())) - Z + t * target_penalty

        N_prime = alpha * (2.0 * t * p + q_eff)
        f_prime_val = ((N_prime * alpha_minus_t2) + 4.0 * t * N_num) / (alpha_minus_t2 ** 3) + target_penalty

        # Damped Newton Step: Use abs() to prevent moving up the gradient in non-convex regions
        raw_step = f_val / (f_prime_val.abs() + damping)
        
        # Clamp the step size to prevent overshooting into the singularity zone
        step = torch.clamp(raw_step, min=-max_step_size, max=max_step_size)
        
        t = t - step

    # Final boundary clamp before analytical reconstruction 
    # (prevents accumulation blow-ups in the next step)
    t = torch.clamp(t, min=-max_t_val, max=max_t_val)

    # 4. Analytical Consensus Reconstruction
    t2 = t.square()
    denom = alpha - t2
    inv_denom = 1.0 / denom       # (M, N) - Safe because denom >= eps
    t_inv_denom = t / denom       # (M, N)

    # A_proj analytically averages over N proposals via Matrix Math
    sum_inv_denom_j = inv_denom.sum(dim=-1, keepdim=True) # (..., M, 1)
    A_proj = (alpha / N) * (A * sum_inv_denom_j + t_inv_denom @ B_eff.transpose(-2, -1))

    # B_proj analytically averages over M proposals via Matrix Math
    sum_inv_denom_i = inv_denom.sum(dim=-2, keepdim=True) # (..., 1, N)
    B_eff_proj = (1.0 / M) * (alpha * B_eff * sum_inv_denom_i + A.transpose(-2, -1) @ t_inv_denom)

    Z_proj = Z - t * target_penalty

    # Revert isomorphism transformation to get the actual B_proj
    if residual:
        B_proj = I - B_eff_proj
    else:
        B_proj = B_eff_proj

    return A_proj, B_proj, Z_proj, t.detach()

# ─── autograd.Function wrappers ───────────────────────────────────────────────
class MatMulProjection(torch.autograd.Function):
    """
    Minimize || A_{new} - A_{old} ||_F^2 + alpha * || B_{new} - B_{old} ||_F^2 + g * || Z_{new} - Z_{old} ||_F^2
    subject to A_{new} @ B_{new} = Z_{new} (or A_{new} - A_{new} @ B_{new} = Z_{new} if residual)
    """
    @staticmethod
    def forward(ctx, A, B, num_steps, alpha, g, omega, proj_cache=None, pairwise=False, residual=True, forward_cache=None):
        ctx.save_for_backward(A, B)
        ctx.alpha = alpha
        ctx.g = g
        ctx.num_steps = num_steps
        ctx.proj_cache = proj_cache  # Store reference to the mutable dictionary
        ctx.forward_cache = forward_cache  # Store reference to the mutable dictionary for forward pass caching
        ctx.omega = omega
        ctx.pairwise = pairwise
        ctx.residual = residual
        
        if residual:
            return (A - A @ B) / omega
        else:
            return (A @ B) / omega

    @staticmethod
    def backward(ctx, Z_target):
        A, B = ctx.saved_tensors
        A_det = A.detach()
        B_det = B.detach()
        Z_det = Z_target.detach()

        # Retrieve the cached t from previous runs if available and not pairwise
        t_init = None
        if not ctx.pairwise and ctx.proj_cache is not None:
            t_init = ctx.proj_cache.get('t')

        if not ctx.pairwise and (A_det.ndim > 2 or B_det.ndim > 2):
            # .clone() breaks the view chain so torch.compile doesn't
            # guard on the original 4D _base strides (which vary per layer)
            A_2d = A_det.reshape(-1, A_det.shape[-1]).clone()
            Z_2d = Z_det.reshape(-1, Z_det.shape[-1]).clone() * ctx.omega
            B_2d = B_det.reshape(-1, B_det.shape[-2], B_det.shape[-1]).mean(dim=0).clone()
            
            t_init_2d = t_init.reshape(Z_2d.shape) if (t_init is not None and t_init.shape == Z_det.shape) else None
            
            A_proj_2d, B_proj_2d, Z_proj_2d, t_new = matmul_proj(
                A_2d, B_2d, Z_2d, t_init=t_init_2d, alpha=ctx.alpha, g=ctx.g, 
                omega=ctx.omega, num_steps=ctx.num_steps, residual=ctx.residual
            )
            
            A_proj = A_proj_2d.reshape(A_det.shape)
            if config.frozen_a_weights:
                B_proj_2d = compute_weight_target_frozen_a(
                    A_2d, B_2d, Z_2d, g=ctx.g, omega=ctx.omega, residual=ctx.residual)
            B_proj_2d = process_weight_target(B_2d, B_proj_2d)
            B_proj = B_proj_2d.reshape(B_det.shape[-2], B_det.shape[-1]).expand(B_det.shape)
            Z_proj = Z_proj_2d.reshape(Z_det.shape)

            if ctx.proj_cache is not None:
                ctx.proj_cache['t'] = t_new.reshape(Z_det.shape)
        else:
            # .contiguous().clone() ensures no view _base for 2D inputs
            # that may come from transpose() or other view ops
            A_proj, B_proj, Z_proj, t_new = matmul_proj(
                A_det.contiguous().clone(), B_det.contiguous().clone(), Z_det.contiguous().clone() * ctx.omega,
                t_init=t_init, alpha=ctx.alpha, g=ctx.g, omega=ctx.omega,
                num_steps=ctx.num_steps, residual=ctx.residual
            )

            # Update the cache for the next iteration if not pairwise
            if not ctx.pairwise and ctx.proj_cache is not None:
                ctx.proj_cache['t'] = t_new

            if config.frozen_a_weights:
                A_2d_fb = A_det.contiguous().clone()
                Z_2d_fb = Z_det.contiguous().clone() * ctx.omega
                B_proj = compute_weight_target_frozen_a(
                    A_2d_fb, B_det.contiguous().clone(), Z_2d_fb,
                    g=ctx.g, omega=ctx.omega, residual=ctx.residual)
            B_proj = process_weight_target(B_det, B_proj)

        if ctx.forward_cache is not None:
            ctx.forward_cache[0] = Z_proj
        return process_activation_target(A_det, A_proj), B_proj, None, None, None, None, None, None, None, None


@torch.compile()
def matmul_proj_fixed_A(A, B, Z, g=1.0, omega=1.0, residual=False):
    """
    Exact projection onto {A, B, Z : A @ B = Z} with A held fixed.
    A: (..., M, K)
    B: (..., K, N)
    Z: (..., M, N)
    """
    M = A.size(-2)
    N = B.size(-1)

    if residual:
        I_res = torch.eye(B.size(-2), B.size(-1), device=B.device, dtype=B.dtype)
        B_eff = I_res - B
    else:
        B_eff = B

    lam = (omega ** 2) / (g ** 2)
    P = A @ B_eff  # (..., M, N)
    Z0 = Z * omega # (..., M, N)

    S = A @ A.transpose(-1, -2) # (..., M, M)
    I = torch.eye(M, device=A.device, dtype=A.dtype)
    system_matrix = I + lam * S

    rhs = P + lam * (S @ Z0)

    # Solve (I + lam*S) Z_proj = rhs
    Z_proj = torch.linalg.solve(system_matrix, rhs)

    T = lam * (Z_proj - Z0)
    B_eff_proj = B_eff - A.transpose(-1, -2) @ T

    Z_proj = Z_proj / omega

    if residual:
        B_proj = I_res - B_eff_proj
    else:
        B_proj = B_eff_proj

    return A, B_proj, Z_proj, None


class MatMulProjectionFrozenA(torch.autograd.Function):
    """
    Minimize || B_{new} - B_{old} ||_F^2 + g * || Z_{new} - Z_{old} ||_F^2
    subject to A_{old} @ B_{new} = Z_{new} (A is held fixed)
    """
    @staticmethod
    def forward(ctx, A, B, g, omega, residual=False, forward_cache=None):
        ctx.save_for_backward(A, B)
        ctx.g = g
        ctx.omega = omega
        ctx.residual = residual
        ctx.forward_cache = forward_cache
        
        if residual:
            return (A - A @ B) / omega
        else:
            return (A @ B) / omega

    @staticmethod
    def backward(ctx, Z_target):
        A, B = ctx.saved_tensors
        A_det = A.detach()
        B_det = B.detach()
        Z_det = Z_target.detach()

        g_frozen = ctx.g * config.frozen_a_g
        A_proj, B_proj, Z_proj, _ = matmul_proj_fixed_A(
            A_det.contiguous().clone(), B_det.contiguous().clone(), Z_det.contiguous().clone() * ctx.omega,
            g=g_frozen, omega=ctx.omega, residual=ctx.residual
        )

        if ctx.forward_cache is not None:
            ctx.forward_cache[0] = Z_proj

        return process_activation_target(A_det, A_proj), process_weight_target(B_det, B_proj), None, None, None, None


class MSEProjection(torch.autograd.Function):
    """
    Minimize || predictions_{new} - predictions_{old} ||_2^2 + ||targets_{new} - targets_{old} ||_2^2 subject to 
    predictions_{new} = targets_{new}
    """
    @staticmethod
    def forward(ctx, predictions, targets):
        ctx.save_for_backward(predictions, targets)
        return F.mse_loss(predictions, targets)

    @staticmethod
    def backward(ctx, z):
        predictions, targets = ctx.saved_tensors
        out = (predictions + targets) / 2
        return out, out

class CrossEntropyProjection(torch.autograd.Function):
    """
    prox_{lambda * l_CE(., y)}(x_0) = arg min_{x} (lambda * l_CE(x, y) + 1/2 * ||x - x_0||^2)
    """
    
    @staticmethod
    def forward(ctx, logits, labels, num_steps=5, lmbda = 1.0):
        ctx.save_for_backward(logits, labels)
        ctx.num_steps = num_steps
        ctx.lmbda = lmbda
        return logits

    @staticmethod
    def backward(ctx, z_target):
        logits, labels = ctx.saved_tensors
        lmbda = ctx.lmbda
        steps = ctx.num_steps

        x = logits
        for _ in range(steps):
            x = x + lmbda * (labels - F.softmax(x, dim=-1))

        #print(f"DEBUG CrossEntropyProjection.backward returning x of shape {x.shape}")
        return x, labels

class HardMarginProjection(torch.autograd.Function):
    """
    Your original implementation: The 'Strict Teacher'.
    Hard-clips logits to the boundary immediately. 
    (Equivalent to the proximal operator of an indicator/constraint function).
    """
    @staticmethod
    def forward(ctx, logits, labels):
        ctx.save_for_backward(logits, labels)
        return logits

    @staticmethod
    def backward(ctx, z_target):
        logits, labels = ctx.saved_tensors
        cond_0 = (labels <= 0) & (logits > 0)
        cond_1 = (labels > 0) & (logits < labels)

        new_logits = logits.clone()
        new_logits[cond_0] = 0.0
        new_logits[cond_1] = labels[cond_1]

        # Note: returning None for labels since we don't need gradients for targets
        return new_logits, None 


class ProximalHingeMargin(torch.autograd.Function):
    """
    Soft Margin Variant 1: The 'Hinge' Proximal Operator.
    Instead of teleporting all the way to the boundary, it takes a bounded step 
    (size lambda_val) towards the correct margin.
    """
    @staticmethod
    def forward(ctx, logits, labels, lambda_val=1.0):
        ctx.save_for_backward(logits, labels)
        ctx.lambda_val = lambda_val
        return logits

    @staticmethod
    def backward(ctx, z_target):
        logits, labels = ctx.saved_tensors
        lmbda = ctx.lambda_val

        cond_0 = (labels <= 0) & (logits > 0)
        cond_1 = (labels > 0) & (logits < labels)

        new_logits = logits.clone()
        
        # Move towards 0 by at most lambda, but don't overshoot past 0
        new_logits[cond_0] = torch.maximum(logits[cond_0] - lmbda, torch.zeros_like(logits[cond_0]))
        
        # Move towards the label by at most lambda, but don't overshoot the label
        new_logits[cond_1] = torch.minimum(logits[cond_1] + lmbda, labels[cond_1])

        return new_logits, None, None


class SmoothSoftMargin(torch.autograd.Function):
    """
    Soft Margin Variant 2: The 'Continuous/Logistic' pull.
    Similar to Cross-Entropy from the paper. It never fully 'gives up', 
    and applies a smooth, continuous force pulling the logit to the correct side.
    """
    @staticmethod
    def forward(ctx, logits, labels, step_size=1.0):
        ctx.save_for_backward(logits, labels)
        ctx.step_size = step_size
        return logits

    @staticmethod
    def backward(ctx, z_target):
        logits, labels = ctx.saved_tensors
        step_size = ctx.step_size

        # Convert labels to standard SVM/Logistic format: -1 for negative, 1 for positive
        y = torch.where(labels > 0, 1.0, -1.0)

        # Gradient of the smooth soft margin loss: log(1 + exp(-y * logits))
        # grad = -y / (1 + exp(y * logits))
        grad = -y / (1.0 + torch.exp(y * logits))

        # Because your custom autograd expects the "projected state" rather than 
        # the raw gradient, we apply the gradient step directly to the logits here.
        new_logits = logits - (step_size * grad)

        return new_logits, None, None


class Conversion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input

    @staticmethod
    def backward(ctx, z_target):
        (input,) = ctx.saved_tensors
        # Convert projection target into gradient: push input toward target
        grad = (input - z_target)
        #grad = grad/ torch.norm(grad)
        return grad

class ReLULInfinityProjection(torch.autograd.Function):
    """
    Sets the new input directly based on the target z,
    projecting onto the ReLU graph using the L-infinity norm.
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.relu(x)

    @staticmethod
    def backward(ctx, z):
        x, = ctx.saved_tensors

        # Solution 1: project onto inactive branch (x <= 0, output = 0)
        x_1 = torch.clamp(x, max=0)
        # L-infinity distance: max(|x - x_1|, |z - 0|)
        dist_1 = torch.max(torch.abs(x - x_1), torch.abs(z))

        # Solution 2: project onto active branch (x > 0, output = x)
        x_2 = torch.clamp((x + z) / 2.0, min=0)
        # L-infinity distance: max(|x - x_2|, |z - x_2|)
        dist_2 = torch.max(torch.abs(x - x_2), torch.abs(z - x_2))

        # Select solution minimizing L-infinity distance
        result = torch.where(dist_1 < dist_2, x_1, x_2)
        return process_activation_target(x, result)
    
class ReLUProjection(torch.autograd.Function):
    """
    Sets the new input directly based on the target z.
    Ignores the original input x entirely.
    """
    @staticmethod
    def forward(ctx, x, forward_cache=None):
        ctx.save_for_backward(x)
        ctx.forward_cache = forward_cache
        return torch.relu(x)

    @staticmethod
    def backward(ctx, z):
        x, = ctx.saved_tensors

        # Solution 1: project onto inactive branch (x <= 0, output = 0)
        x_1 = torch.clamp(x, max=0)
        dist_1 = (x - x_1)**2 + z**2

        # Solution 2: project onto active branch (x > 0, output = x)
        x_2 = torch.clamp((x + z) / 2.0, min=0)
        dist_2 = (x - x_2)**2 + (z - x_2)**2

        # Select solution minimizing distance
        result_backwards = torch.where(dist_1 < dist_2, x_1, x_2)
        result_forwards = torch.where(dist_1 < dist_2, torch.zeros_like(x), x_2)
        ctx.forward_cache[0] = result_forwards        
        return process_activation_target(x, result_backwards), None    


class LeakyReLUProjection(torch.autograd.Function):
    """
    Projection-aware LeakyReLU.

    Forward: y = max(x, negative_slope * x)
    Backward: projects x onto the closest branch-consistent point given z.
    """
    @staticmethod
    def forward(ctx, x, negative_slope=0.01):
        ctx.save_for_backward(x)
        ctx.negative_slope = float(negative_slope)
        return F.leaky_relu(x, negative_slope=ctx.negative_slope)

    @staticmethod
    def backward(ctx, z):
        (x,) = ctx.saved_tensors
        slope = ctx.negative_slope

        # Branch 1 (inactive): y = slope * x, x <= 0
        x_1_unconstrained = (x + slope * z) / (1.0 + slope * slope)
        x_1 = torch.clamp(x_1_unconstrained, max=0)
        y_1 = slope * x_1
        dist_1 = (x - x_1) ** 2 + (z - y_1) ** 2

        # Branch 2 (active): y = x, x >= 0
        x_2_unconstrained = (x + z) / 2.0
        x_2 = torch.clamp(x_2_unconstrained, min=0)
        y_2 = x_2
        dist_2 = (x - x_2) ** 2 + (z - y_2) ** 2

        result = torch.where(dist_1 < dist_2, x_1, x_2)
        return process_activation_target(x, result), None


class SelectTokenProjection(torch.autograd.Function):
    """
    Projection-aware token selection (e.g. CLS token extraction).
    Forward: returns x[:, index].
    Backward: places the target at position `index` and keeps all other
    positions at their original values (identity), so upstream projections
    don't get a zero-target for non-selected positions.
    """
    @staticmethod
    def forward(ctx, x, index):
        ctx.save_for_backward(x)
        ctx.index = index
        return x[:, index]           # (B, emb_dim)

    @staticmethod
    def backward(ctx, y_target):
        x, = ctx.saved_tensors
        # Start from the original values (identity for non-selected positions)
        x_star = x.clone()
        # Only the selected position gets the projection target
        x_star[:, ctx.index] = y_target
        return x_star, None          # None for `index`
    
class MeanProjection(torch.autograd.Function):
    """
    Projects inputs onto the mean constraint graph.
    https://gemini.google.com/share/53aeb9b49de3
    """
    @staticmethod
    def forward(ctx, x, dim=-1):
        # Note: Changed *inputs to x as torch.mean operates on a single tensor.
        ctx.save_for_backward(x)
        ctx.dim = dim
        return torch.mean(x, dim=dim)
    
    @staticmethod
    def backward(ctx, z_target):
        x, = ctx.saved_tensors
        dim = ctx.dim
        n = x.size(dim)
        
        # Calculate the mean of the original input
        x_mean = torch.mean(x, dim=dim, keepdim=True)
        
        # Align z_target dimensions for broadcasting if the forward pass reduced the dimension
        if z_target.dim() < x.dim():
            z_target_expanded = z_target.unsqueeze(dim)
        else:
            z_target_expanded = z_target
            
        # Compute lambda: (y_0 - mean(x_0)) / (n + 1)
        lambda_val = (z_target_expanded - x_mean) / (n + 1)
        
        # Projected input: x_0 + lambda * 1
        projected_x = x + lambda_val
        
        # Return the projected tensor. 
        # We must return None for the 'dim' argument since it does not require a gradient.
        return projected_x, None


class StepProjection(torch.autograd.Function):
    """
    PyTorch equivalent of PJAX step activation projection.
    Forward: step(sum of inputs) -> 1 if sum >= 0 else -1
    Backward: projects inputs onto the step constraint graph.
    """
    @staticmethod
    def forward(ctx, *inputs):
        ctx.save_for_backward(*inputs)
        s = sum(inputs)
        return torch.where(s >= 0, torch.tensor(1.0, dtype=s.dtype, device=s.device), 
                           torch.tensor(-1.0, dtype=s.dtype, device=s.device))

    @staticmethod
    def backward(ctx, z_target):
        inputs = ctx.saved_tensors
        n = len(inputs)
        
        s = sum(inputs)
        
        # Midpoint adjustment: move inputs toward the target output
        # (s - output) / (len(inputs) + 1)
        mid = (s - z_target) / (n + 1)
        
        return projected_inputs
class QuantizeReLUProjection(torch.autograd.Function):
    """
    Quantized ReLU with Projection-aware gradients.
    
    Forward: 
        f(x) = step * round(max(0, x) / step)
    
    Backward: 
        Projects (x, z_target) onto the nearest segment of the non-negative 
        staircase. The first segment (k=0) extends from (-inf, 0.5 * step].
    """
    @staticmethod
    def forward(ctx, x, step=1.0):
        ctx.save_for_backward(x)
        ctx.step = step
        # Apply ReLU then round to nearest step
        return step * torch.round(torch.clamp(x, min=0.0) / step)

    @staticmethod
    def backward(ctx, z_target):
        x, = ctx.saved_tensors
        s = ctx.step

        # Nearest rung index to target, but forced to be at least 0 (ReLU)
        k0 = torch.clamp(torch.round(z_target / s), min=0.0)

        best_x = x.clone()
        best_dist = torch.full_like(x, float('inf'))

        # Check candidate levels: k0-1, k0, k0+1
        for dk in (-1, 0, 1):
            k = k0 + dk
            
            valid_mask = k >= 0
            
            level = k * s
            
            # For k=0, the domain is (-inf, 0.5 * s]
            # For k>0, the domain is [(k - 0.5) * s, (k + 0.5) * s]
            lo = torch.where(k == 0, torch.full_like(k, float('-inf')), (k - 0.5) * s)
            hi = (k + 0.5) * s
            
            x_clamped = torch.maximum(lo, torch.minimum(x, hi))
            dist = (x - x_clamped) ** 2 + (z_target - level) ** 2
            
            better = valid_mask & (dist < best_dist)
            best_x = torch.where(better, x_clamped, best_x)
            best_dist = torch.where(better, dist, best_dist)

        # Assuming process_activation_target handles the STE or projection update
        return process_activation_target(x, best_x), None

class GapProjection(torch.autograd.Function):
    """
    Projection-aware Gap activation.

    Forward: identity (pass-through).
    Backward: projects input x onto the feasible set |x| >= delta/2,
    enforcing a gap of width `delta` in output space centred at zero.
    Values inside the gap are snapped to the nearest boundary (+delta/2
    or -delta/2), forcing the network to commit.
    """
    @staticmethod
    def forward(ctx, x, delta=2.0):
        ctx.save_for_backward(x)
        ctx.delta = delta
        return x

    @staticmethod
    def backward(ctx, z_target):
        x, = ctx.saved_tensors
        half = ctx.delta / 2.0

        # Project: snap values inside the gap to the nearest boundary
        result = torch.where(
            z_target >= 0,
            torch.clamp(z_target, min=half),
            torch.clamp(z_target, max=-half),
        )

        return process_activation_target(x, result), None

class GappedStepProjection(torch.autograd.Function):
    """
    Projection-aware Gapped Step activation.

    Like the regular Step activation but with a dead zone of width `delta`
    centred at the origin where the function is undefined:

        f(x) = +1   if x >= delta/2
        f(x) = -1   if x <= -delta/2
        (undefined)  if -delta/2 < x < delta/2

    The projection selects the closer of the two feasible branches:
        Branch 1 (positive): x' = clamp(x, min=delta/2), y = +1
        Branch 2 (negative): x' = clamp(x, max=-delta/2), y = -1
    """
    @staticmethod
    def forward(ctx, x, delta=2.0):
        ctx.save_for_backward(x)
        ctx.delta = delta
        half = delta / 2.0
        return torch.where(x >= half, torch.ones_like(x),
               torch.where(x <= -half, -torch.ones_like(x),
                            torch.zeros_like(x)))

    @staticmethod
    def backward(ctx, z_target):
        x, = ctx.saved_tensors
        half = ctx.delta / 2.0

        # Branch 1: positive (x >= delta/2, y = +1)
        x_pos = torch.clamp(x, min=half)
        dist_pos = (x - x_pos) ** 2 + (z_target - 1.0) ** 2

        # Branch 2: negative (x <= -delta/2, y = -1)
        x_neg = torch.clamp(x, max=-half)
        dist_neg = (x - x_neg) ** 2 + (z_target + 1.0) ** 2

        # Select branch with minimum distance
        result = torch.where(dist_pos <= dist_neg, x_pos, x_neg)

        return process_activation_target(x, result), None

class DropoutProjection(torch.autograd.Function):
    """
    Projection-aware dropout.

    Forward:  z = x * mask / (1 - p)   (standard inverted dropout)
    Backward: projects (x, z) onto the dropout function graph.

    For *dropped* positions (mask=0) the constraint is z=0 regardless of x,
    so x is kept at its original value (identity projection).

    For *kept* positions the constraint is z = x / (1-p).  Setting s = 1/(1-p):
        min  ||x - x0||^2 + ||z - z0||^2   s.t. z = s*x
        =>   x* = (x0 + s*z0) / (1 + s^2)
             z* = s * x*
    """
    @staticmethod
    def forward(ctx, x, p=0.5, training=True):
        if training and p > 0.0:
            mask = (torch.rand_like(x) > p).to(x.dtype)
            scale = 1.0 / (1.0 - p)
            y = x * mask * scale
        else:
            mask = torch.ones_like(x)
            scale = 1.0
            y = x

        ctx.save_for_backward(x, mask)
        ctx.p = p
        ctx.scale = scale
        ctx.training = training
        return y

    @staticmethod
    def backward(ctx, z_target):
        x, mask = ctx.saved_tensors
        s = ctx.scale  # 1 / (1 - p)

        # Kept positions: project onto z = s*x
        x_proj = (x + s * z_target) / (1.0 + s * s)

        # Dropped positions: x stays at original (output is forced to 0)
        x_star = torch.where(mask > 0, x_proj, x)

        # Return None for p, training (non-differentiable args)
        return x_star, None, None

