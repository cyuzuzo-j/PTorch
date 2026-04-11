import torch
import torch.nn.functional as F
import warnings
from .. import config
import math

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    is_1d = G.ndim == 1
    if is_1d:
        G = G.view(1, -1)
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    out = X.T if transposed else X
    if is_1d:
        out = out.view(-1)
    return out.to(G.dtype)

def process_activation_target(A_det, A_proj):
    if getattr(config, 'muon_activations', True):
        orig_shape = A_det.shape
        A_2d = A_det.reshape(-1, orig_shape[-1])
        A_proj_2d = A_proj.reshape(-1, orig_shape[-1])
        g = A_2d - A_proj_2d
        
        g_muon = zeropower_via_newtonschulz5(g)
        lr = getattr(config, 'muon_activations_lr', 1.0)
        
        if getattr(config, 'muon_activations_scale', False):
            scale = g.norm() / (g_muon.norm() + 1e-8)
            g_muon = g_muon * scale
            
        A_proj_new = A_2d - lr * g_muon
        return A_proj_new.reshape(orig_shape)
    return A_proj

# Suppress pin_memory warning when no accelerator is available
warnings.filterwarnings("ignore", message=".*pin_memory.*")

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
        

# @torch.compile(dynamic=True)
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
# @torch.compile(dynamic=True)
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
    def forward(ctx, A, B, num_steps, alpha, g, omega, proj_cache=None, pairwise=False, residual=True):
        ctx.save_for_backward(A, B)
        ctx.alpha = alpha
        ctx.g = g
        ctx.num_steps = num_steps
        ctx.proj_cache = proj_cache  # Store reference to the mutable dictionary
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
            
            A_proj_2d, B_proj_2d, _, t_new = matmul_proj(
                A_2d, B_2d, Z_2d, t_init=t_init_2d, alpha=ctx.alpha, g=ctx.g, 
                omega=ctx.omega, num_steps=ctx.num_steps, residual=ctx.residual
            )
            
            A_proj = A_proj_2d.reshape(A_det.shape)
            B_proj = B_proj_2d.reshape(B_det.shape[-2], B_det.shape[-1]).expand(B_det.shape)
            
            if ctx.proj_cache is not None:
                ctx.proj_cache['t'] = t_new.reshape(Z_det.shape)
        else:
            # .contiguous().clone() ensures no view _base for 2D inputs
            # that may come from transpose() or other view ops
            A_proj, B_proj, _, t_new = matmul_proj(
                A_det.contiguous().clone(), B_det.contiguous().clone(), Z_det.contiguous().clone() * ctx.omega,
                t_init=t_init, alpha=ctx.alpha, g=ctx.g, omega=ctx.omega, 
                num_steps=ctx.num_steps, residual=ctx.residual
            )

            # Update the cache for the next iteration if not pairwise
            if not ctx.pairwise and ctx.proj_cache is not None:
                ctx.proj_cache['t'] = t_new

        # Return Nones for proj_cache, pairwise, and residual 
        return process_activation_target(A_det, A_proj), B_proj, None, None, None, None, None, None, None


class MatMulProjectionDTP(torch.autograd.Function):
    """
    Minimize || A_{new} - A_{old} ||_F^2 + alpha * || B_{new} - B_{old} ||_F^2 + g * || Z_{new} - Z_{old} ||_F^2
    subject to A_{new} @ B_{new} = Z_{new} (or A_{new} - A_{new} @ B_{new} = Z_{new} if residual)
    
    Includes a Difference Target Propagation (DTP) style correction term:
    Target_A = Proj(Z_target)_A + (A_orig - Proj(Z_orig)_A)
    """
    @staticmethod
    def forward(ctx, A, B, num_steps, alpha, g, omega, proj_cache=None, pairwise=False, residual=True):
        ctx.save_for_backward(A, B)
        ctx.alpha = alpha
        ctx.g = g
        ctx.num_steps = num_steps
        ctx.proj_cache = proj_cache  # Store reference to the mutable dictionary
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
            B_2d = B_det.reshape(-1, B_det.shape[-2], B_det.shape[-1]).mean(dim=0).clone()
            
            # 1. Calculate the original (unscaled) output Z_orig
            if ctx.residual:
                Z_orig_2d = A_2d - A_2d @ B_2d
            else:
                Z_orig_2d = A_2d @ B_2d
                
            # 2. Reconstruct A and B from the original output (Inverse mapping G(h_{i+1}))
            # We pass t_init=None here so we don't corrupt the momentum state meant for the actual target
            A_recon_2d, B_recon_2d, _, _ = matmul_proj(
                A_2d, B_2d, Z_orig_2d, t_init=None, alpha=ctx.alpha, g=ctx.g, 
                omega=ctx.omega, num_steps=ctx.num_steps, residual=ctx.residual
            )
            
            # 3. Calculate DTP Calibration Offsets (h_i - G(h_{i+1}))
            offset_A_2d = A_2d - A_recon_2d
            offset_B_2d = B_2d - B_recon_2d

            # 4. Project the actual target (G(t_{i+1}))
            Z_target_2d = Z_det.reshape(-1, Z_det.shape[-1]).clone() * ctx.omega
            t_init_2d = t_init.reshape(Z_target_2d.shape) if (t_init is not None and t_init.shape == Z_det.shape) else None
            
            A_proj_2d, B_proj_2d, _, t_new = matmul_proj(
                A_2d, B_2d, Z_target_2d, t_init=t_init_2d, alpha=ctx.alpha, g=ctx.g, 
                omega=ctx.omega, num_steps=ctx.num_steps, residual=ctx.residual
            )
            
            # 5. Apply the DTP Correction
            A_proj_2d = A_proj_2d + offset_A_2d
            B_proj_2d = B_proj_2d + offset_B_2d
            
            # Reshape back to original dimensions
            A_proj = A_proj_2d.reshape(A_det.shape)
            B_proj = B_proj_2d.reshape(B_det.shape[-2], B_det.shape[-1]).expand(B_det.shape)
            
            if ctx.proj_cache is not None:
                ctx.proj_cache['t'] = t_new.reshape(Z_det.shape)
                
        else:
            # contiguous().clone() ensures no view _base for 2D inputs
            A_clone = A_det.contiguous().clone()
            B_clone = B_det.contiguous().clone()
            
            # 1. Calculate the original (unscaled) output Z_orig
            if ctx.residual:
                Z_orig = A_clone - A_clone @ B_clone
            else:
                Z_orig = A_clone @ B_clone
                
            # 2. Reconstruct A and B from the original output
            A_recon, B_recon, _, _ = matmul_proj(
                A_clone, B_clone, Z_orig, t_init=None, alpha=ctx.alpha, g=ctx.g, 
                omega=ctx.omega, num_steps=ctx.num_steps, residual=ctx.residual
            )
            
            # 3. Calculate DTP Calibration Offsets
            offset_A = A_clone - A_recon
            offset_B = B_clone - B_recon

            # 4. Project the actual target
            Z_target_scaled = Z_det.contiguous().clone() * ctx.omega
            A_proj, B_proj, _, t_new = matmul_proj(
                A_clone, B_clone, Z_target_scaled,
                t_init=t_init, alpha=ctx.alpha, g=ctx.g, omega=ctx.omega, 
                num_steps=ctx.num_steps, residual=ctx.residual
            )
            
            # 5. Apply the DTP Correction
            A_proj = A_proj + offset_A
            B_proj = B_proj + offset_B

            # Update the cache for the next iteration if not pairwise
            if not ctx.pairwise and ctx.proj_cache is not None:
                ctx.proj_cache['t'] = t_new

        # Return Nones for num_steps, alpha, g, omega, proj_cache, pairwise, and residual 
        return process_activation_target(A_det, A_proj), B_proj, None, None, None, None, None, None, None


# @torch.compile(dynamic=True)
def matmul_proj_l1(A, B, Z, eps_init=None, g=1.0, omega=1.0, num_steps=None, residual=False):
    """
    Exact independent bilinear projection for A @ B = Z using the L_1 norm.
    
    Analytically optimized. Unlike L_inf, L_1 projection promotes extreme sparsity. 
    For any (m, n) pair, the optimal projection will modify exactly ONE feature 
    dimension k* (or just update Z if the g-penalty is cheap enough). 
    This allows us to solve it exactly in zero iterations.
    """
    M, K = A.size(-2), A.size(-1)
    N = B.size(-1)

    if residual:
        I = torch.eye(B.size(-2), B.size(-1), device=B.device, dtype=B.dtype)
        B_eff = I - B
    else:
        B_eff = B

    # 1. Base pairwise projection & residual
    P = A @ B_eff  # (..., M, N)
    Z_target = Z * omega
    Delta = Z_target - P  # (..., M, N)

    # 2. Expand arrays to (..., M, N, K) to test all single-coordinate hypotheses
    A_mnk = A.unsqueeze(-2).expand(*A.shape[:-2], M, N, K)
    B_mnk = B_eff.transpose(-1, -2).unsqueeze(-3).expand(*B_eff.shape[:-2], M, N, K)
    
    # Target product c_k if dimension k were to absorb the entire delta
    c = A_mnk * B_mnk + Delta.unsqueeze(-1)

    # 3. Formulate the 4 geometric intersection candidates for the L1 diamond
    eps_div = 1e-12 # Prevent zero-division on axes
    
    # Candidate 1: Vertex A
    x1 = A_mnk
    y1 = torch.where(A_mnk.abs() < eps_div, c / eps_div, c / A_mnk)
    cost1 = torch.abs(y1 - B_mnk)

    # Candidate 2: Vertex B
    x2 = torch.where(B_mnk.abs() < eps_div, c / eps_div, c / B_mnk)
    y2 = B_mnk
    cost2 = torch.abs(x2 - A_mnk)

    # Candidates 3 & 4: Flat Tangents (|x| = |y|)
    sqrt_c = torch.sqrt(torch.abs(c))
    x3 = torch.sign(c) * sqrt_c
    y3 = sqrt_c
    cost3 = torch.abs(x3 - A_mnk) + torch.abs(y3 - B_mnk)

    x4 = -x3
    y4 = -y3
    cost4 = torch.abs(x4 - A_mnk) + torch.abs(y4 - B_mnk)

    # 4. Find the global minimum cost out of the 4*K options + 1 Z-slack option
    costs = torch.stack([cost1, cost2, cost3, cost4], dim=-1) # (..., M, N, K, 4)
    costs_flat = costs.reshape(*costs.shape[:-2], K * 4)      # (..., M, N, K*4)
    
    Z_cost = ((g / omega) * torch.abs(Delta)).unsqueeze(-1)   # (..., M, N, 1)
    all_costs = torch.cat([costs_flat, Z_cost], dim=-1)       # (..., M, N, K*4 + 1)
    
    best_idx = torch.argmin(all_costs, dim=-1)                # (..., M, N)

    # 5. Extract the winning coordinates
    K4 = K * 4
    is_Z_update = (best_idx == K4)
    
    # Clamp to avoid gather out-of-bounds for the Z_update indices
    safe_best_idx = torch.clamp(best_idx, max=K4 - 1).unsqueeze(-1)
    safe_k_idx = torch.div(torch.clamp(best_idx, max=K4 - 1), 4, rounding_mode='floor').unsqueeze(-1)

    x_cands = torch.stack([x1, x2, x3, x4], dim=-1).reshape(*costs.shape[:-2], K * 4)
    y_cands = torch.stack([y1, y2, y3, y4], dim=-1).reshape(*costs.shape[:-2], K * 4)
    
    best_x = torch.gather(x_cands, -1, safe_best_idx).squeeze(-1) # (..., M, N)
    best_y = torch.gather(y_cands, -1, safe_best_idx).squeeze(-1) # (..., M, N)

    orig_A = torch.gather(A_mnk, -1, safe_k_idx).squeeze(-1)
    orig_B = torch.gather(B_mnk, -1, safe_k_idx).squeeze(-1)

    # Nullify deltas if Z took the penalty
    delta_A = torch.where(is_Z_update, 0.0, best_x - orig_A)
    delta_B = torch.where(is_Z_update, 0.0, best_y - orig_B)

    # Scatter deltas back to their sparse feature dimension
    dA = torch.zeros_like(A_mnk)
    dA.scatter_(-1, safe_k_idx, delta_A.unsqueeze(-1))
    
    dB = torch.zeros_like(B_mnk)
    dB.scatter_(-1, safe_k_idx, delta_B.unsqueeze(-1))

    # 6. Consensus Reconstruction
    A_proj = A + dA.mean(dim=-2) 
    B_eff_proj = B_eff + dB.mean(dim=-3).transpose(-1, -2) 
    
    # If Z won the local optimization, it yields to P; otherwise it holds steady
    Z_proj = torch.where(is_Z_update, P / omega, Z)

    if residual:
        B_proj = I - B_eff_proj
    else:
        B_proj = B_eff_proj
        
    # We return a zero tensor for eps_new to match the tuple signature, 
    # as eps has no meaning in L_1's discrete argmin bounds.
    return A_proj, B_proj, Z_proj, torch.zeros_like(P)

class MatMulProjectionL1(torch.autograd.Function):
    """
    Minimize || A_{new} - A_{old} ||_1 + || B_{new} - B_{old} ||_1 + g * || Z_{new} - Z_{old} ||_1
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
            
            A_proj_2d, B_proj_2d, _, eps_new = matmul_proj_l1(
                A_2d, B_2d, Z_2d, eps_init=eps_init_2d, g=ctx.g, 
                omega=ctx.omega, num_steps=ctx.num_steps, residual=ctx.residual
            )
            
            A_proj = A_proj_2d.reshape(A_det.shape)
            B_proj = B_proj_2d.reshape(B_det.shape[-2], B_det.shape[-1]).expand(B_det.shape)
            
            if ctx.proj_cache is not None:
                ctx.proj_cache['eps'] = eps_new.reshape(Z_det.shape)
        else:
            A_proj, B_proj, _, eps_new = matmul_proj_l1(
                A_det.contiguous().clone(), B_det.contiguous().clone(), Z_det.contiguous().clone() * ctx.omega,
                eps_init=eps_init, g=ctx.g, omega=ctx.omega, 
                num_steps=ctx.num_steps, residual=ctx.residual
            )

            if not ctx.pairwise and ctx.proj_cache is not None:
                ctx.proj_cache['eps'] = eps_new

        # Note: the alpha scalar weighting is intentionally removed for L_1
        return process_activation_target(A_det, A_proj), B_proj, None, None, None, None, None, None


# @torch.compile(dynamic=True)
def matmul_proj_lp(A, B, Z, p, u_init=None, g=1.0, omega=1.0, num_steps=10, residual=False):
    """
    Exact independent bilinear projection for A @ B = Z using a general L_p norm (1 < p < inf).

    Implements Algorithm §7 from the LP projection derivation:
      - Joint Newton–Raphson root-finding over V = (x_i, w_i, u) per (m, n) pair
      - O(K) per-iteration via Schur complement on the block-diagonal Jacobian
      - Three numerical safeguards: clamped phi_q' (§6.3), Tikhonov (§6.4), Armijo (§6.5)

    Args:
        A:         Input activations       (..., M, K)
        B:         Weight matrix            (..., K, N)   [or (K, N)]
        Z:         Target output            (..., M, N)
        p:         Norm exponent            (float, 1 < p < inf)
        u_init:    Warm-start for dual variable u  (..., M, N) or None
        g:         Penalty weight on Z-deviation
        omega:     Output scale divisor
        num_steps: Maximum Newton iterations
        residual:  If True, constraint is A - A@B = Z

    Returns:
        A_proj, B_proj, Z_proj, u_state
    """
    q = p / (p - 1.0)       # Hölder conjugate
    gamma_q = g ** q         # precomputed γ^q for the output penalty

    if residual:
        I = torch.eye(B.size(-2), B.size(-1), device=B.device, dtype=B.dtype)
        B_eff = I - B
    else:
        B_eff = B

    # Compute residual: how far the current (A, B, Z) is from feasibility
    P = A @ B_eff                          # (..., M, N)
    Z_target = Z * omega
    r0 = Z_target - P                      # initial constraint violation per (m, n)

    # ── Trivial check: if already feasible, skip ─────────────────────────
    # (handled per-element via Newton converging in 0 steps naturally)

    M, K = A.shape[-2], A.shape[-1]
    N = B_eff.shape[-1]

    # Working copies of x (A-features) and w (B_eff-features)
    # Shape: (..., M, K, N) — one (x_i, w_i) pair per output element (m, n)
    x = A.unsqueeze(-1).expand(*A.shape, N).clone()         # (..., M, K, N)
    w = B_eff.unsqueeze(-3).expand(*A.shape[:-1], K, N).clone()  # (..., M, K, N)

    # Dual variable u: one scalar per (m, n) pair
    if u_init is not None and u_init.shape[-2:] == (M, N):
        u = u_init.to(A.device).clone()
    else:
        u = torch.zeros(*A.shape[:-1], N, device=A.device, dtype=A.dtype)  # (..., M, N)

    # Original values (constant)
    x0 = A.unsqueeze(-1).expand_as(x)          # (..., M, K, N)
    w0 = B_eff.unsqueeze(-3).expand_as(w)      # (..., M, K, N)
    y0 = Z_target                               # (..., M, N)

    # Safeguard parameters
    eps_clamp = 1e-8     # §6.3: clamping threshold for phi_q' when p > 2
    eps_det = 1e-6       # §6.4: Tikhonov trigger threshold
    beta_ls = 0.5        # §6.5: Armijo backtracking factor
    mu_ls = 1e-4         # §6.5: Armijo sufficient decrease parameter
    max_ls = 8           # max backtracking steps

    need_clamp = (p > 2.0)   # phi_q' blows up near zero only when q < 2

    for _step in range(num_steps):
        u_k = u.unsqueeze(-2)  # (..., M, 1, N)

        # ── Residuals E_i, G_i, H ────────────────────────────────────────
        # phi_q(z) = sign(z) * |z|^{q-1}
        dx = x - x0          # (..., M, K, N)
        dw = w - w0          # (..., M, K, N)

        phi_q_dx = torch.sign(dx) * torch.abs(dx).pow(q - 1.0)
        phi_q_dw = torch.sign(dw) * torch.abs(dw).pow(q - 1.0)
        phi_q_w = torch.sign(w) * torch.abs(w).pow(q - 1.0)
        phi_q_x = torch.sign(x) * torch.abs(x).pow(q - 1.0)

        # Stationarity residuals (should be zero at solution)
        E = dx - u_k * phi_q_w        # (..., M, K, N)
        G = dw - u_k * phi_q_x        # (..., M, K, N)

        # Constraint residual
        xw_prod = (x * w).sum(dim=-2)  # (..., M, N)
        y_from_u = y0 - u / gamma_q    # (..., M, N)
        H = y0 - u / gamma_q - xw_prod  # (..., M, N)

        # ── Convergence check (per element) ───────────────────────────────
        # merit = sum(E^2 + G^2) + H^2
        merit = (E * E + G * G).sum(dim=-2) + H * H   # (..., M, N)

        # ── Jacobian blocks D_i, b_i, c_i ────────────────────────────────
        # phi_q'(z) = (q-1) * |z|^{q-2}
        if need_clamp:
            # §6.3: Clamp |z| away from zero in the Jacobian only
            abs_x_safe = torch.clamp(torch.abs(x), min=eps_clamp)
            abs_w_safe = torch.clamp(torch.abs(w), min=eps_clamp)
            phi_q_prime_x = (q - 1.0) * abs_x_safe.pow(q - 2.0)
            phi_q_prime_w = (q - 1.0) * abs_w_safe.pow(q - 2.0)
        else:
            phi_q_prime_x = (q - 1.0) * torch.abs(x).pow(q - 2.0)
            phi_q_prime_w = (q - 1.0) * torch.abs(w).pow(q - 2.0)

        # D_i = [[1, -u*phi_q'(w_i)], [-u*phi_q'(x_i), 1]]
        # det(D_i) = 1 - u^2 * phi_q'(x_i) * phi_q'(w_i)
        off_diag_xw = u_k * phi_q_prime_x * phi_q_prime_w * u_k  # u^2 * phi_q'(x) * phi_q'(w)
        det_D = 1.0 - off_diag_xw                                  # (..., M, K, N)

        # §6.4: Tikhonov regularization for near-singular blocks
        needs_reg = (torch.abs(det_D) < eps_det)
        delta = torch.sqrt(torch.abs(off_diag_xw) + eps_det) - 1.0
        delta = torch.clamp(delta, min=0.0)

        # Apply shift: D_i -> D_i + delta_i * I_2
        # New diagonal entries: 1 + delta
        d11 = torch.where(needs_reg, 1.0 + delta, torch.ones_like(delta))
        d22 = d11
        d12 = -u_k * phi_q_prime_w   # (..., M, K, N)
        d21 = -u_k * phi_q_prime_x   # (..., M, K, N)
        det_D_reg = torch.where(needs_reg, d11 * d22 - d12 * d21, det_D)
        inv_det = 1.0 / (det_D_reg + 1e-12)

        # D_i^{-1} = (1/det) * [[d22, -d12], [-d21, d11]]
        # b_i = [-phi_q(w_i), -phi_q(x_i)]
        b1 = -phi_q_w   # (..., M, K, N)
        b2 = -phi_q_x   # (..., M, K, N)

        # c_i = [-w_i, -x_i]
        c1 = -w          # (..., M, K, N)
        c2 = -x          # (..., M, K, N)

        # k = -1/gamma_q
        k = -1.0 / gamma_q

        # ── Schur complement accumulation ─────────────────────────────────
        # Per-block: D_i^{-1} * [E_i, G_i]
        inv_E = inv_det * (d22 * E - d12 * G)    # (..., M, K, N)
        inv_G = inv_det * (-d21 * E + d11 * G)   # (..., M, K, N)

        # Per-block: D_i^{-1} * b_i
        inv_b1 = inv_det * (d22 * b1 - d12 * b2)  # (..., M, K, N)
        inv_b2 = inv_det * (-d21 * b1 + d11 * b2) # (..., M, K, N)

        # c_i^T D_i^{-1} b_i  (summed over K)
        cT_inv_b = (c1 * inv_b1 + c2 * inv_b2).sum(dim=-2)   # (..., M, N)

        # c_i^T D_i^{-1} [E_i, G_i]  (summed over K)
        cT_inv_EG = (c1 * inv_E + c2 * inv_G).sum(dim=-2)    # (..., M, N)

        # Schur complement S = k - sum_i c_i^T D_i^{-1} b_i
        S = k - cT_inv_b   # (..., M, N)

        # ── Multiplier update ─────────────────────────────────────────────
        delta_u = (-H + cT_inv_EG) / (S + 1e-12)              # (..., M, N)

        # ── Variable updates ──────────────────────────────────────────────
        delta_u_k = delta_u.unsqueeze(-2)  # (..., M, 1, N)
        delta_x = -(inv_E + inv_b1 * delta_u_k)  # (..., M, K, N)
        delta_w = -(inv_G + inv_b2 * delta_u_k)  # (..., M, K, N)

        # ── §6.5: Armijo backtracking line search ─────────────────────────
        alpha_step = 1.0
        for _ls in range(max_ls):
            x_try = x + alpha_step * delta_x
            w_try = w + alpha_step * delta_w
            u_try = u + alpha_step * delta_u

            # Recompute merit at trial point
            dx_try = x_try - x0
            dw_try = w_try - w0
            u_try_k = u_try.unsqueeze(-2)

            phi_q_w_try = torch.sign(w_try) * torch.abs(w_try).pow(q - 1.0)
            phi_q_x_try = torch.sign(x_try) * torch.abs(x_try).pow(q - 1.0)

            E_try = dx_try - u_try_k * phi_q_w_try
            G_try = dw_try - u_try_k * phi_q_x_try
            xw_try = (x_try * w_try).sum(dim=-2)
            H_try = y0 - u_try / gamma_q - xw_try

            merit_try = (E_try * E_try + G_try * G_try).sum(dim=-2) + H_try * H_try

            # Accept if sufficient decrease (Armijo condition) on average
            if (merit_try.mean() <= (1.0 - mu_ls * alpha_step) * merit.mean()):
                break
            alpha_step *= beta_ls

        # Apply the accepted step
        x = x + alpha_step * delta_x
        w = w + alpha_step * delta_w
        u = u + alpha_step * delta_u

    # ── Consensus reconstruction ──────────────────────────────────────────
    # Average proposals across output dimension (N) for A, across input dimension (M) for B
    A_proj = x.mean(dim=-1)                    # (..., M, K) — average over N proposals
    B_eff_proj = w.mean(dim=-3)                # (..., K, N) — average over M proposals
    Z_proj = y0 - u / gamma_q                  # (..., M, N)
    Z_proj = Z_proj / omega                    # un-scale

    if residual:
        B_proj = I - B_eff_proj
    else:
        B_proj = B_eff_proj

    return A_proj, B_proj, Z_proj, u.detach()


class MatMulProjectionLp(torch.autograd.Function):
    """
    Minimize sum|A-A0|^p + sum|B-B0|^p + g^p * |Z-Z0|^p
    subject to A @ B = Z   (or A - A@B = Z if residual)

    General L_p norm projection for 1 < p < inf.
    """
    @staticmethod
    def forward(ctx, A, B, num_steps, p, g, omega, proj_cache=None, pairwise=False, residual=True):
        ctx.save_for_backward(A, B)
        ctx.p = p
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

        u_init = None
        if not ctx.pairwise and ctx.proj_cache is not None:
            u_init = ctx.proj_cache.get('u_lp')

        if not ctx.pairwise and (A_det.ndim > 2 or B_det.ndim > 2):
            A_2d = A_det.reshape(-1, A_det.shape[-1]).clone()
            Z_2d = Z_det.reshape(-1, Z_det.shape[-1]).clone() * ctx.omega
            B_2d = B_det.reshape(-1, B_det.shape[-2], B_det.shape[-1]).mean(dim=0).clone()

            u_init_2d = u_init.reshape(Z_2d.shape) if (u_init is not None and u_init.shape == Z_det.shape) else None

            A_proj_2d, B_proj_2d, _, u_new = matmul_proj_lp(
                A_2d, B_2d, Z_2d, p=ctx.p, u_init=u_init_2d, g=ctx.g,
                omega=ctx.omega, num_steps=ctx.num_steps, residual=ctx.residual
            )

            A_proj = A_proj_2d.reshape(A_det.shape)
            B_proj = B_proj_2d.reshape(B_det.shape[-2], B_det.shape[-1]).expand(B_det.shape)

            if ctx.proj_cache is not None:
                ctx.proj_cache['u_lp'] = u_new.reshape(Z_det.shape)
        else:
            A_proj, B_proj, _, u_new = matmul_proj_lp(
                A_det.contiguous().clone(), B_det.contiguous().clone(), Z_det.contiguous().clone() * ctx.omega,
                p=ctx.p, u_init=u_init, g=ctx.g, omega=ctx.omega,
                num_steps=ctx.num_steps, residual=ctx.residual
            )

            if not ctx.pairwise and ctx.proj_cache is not None:
                ctx.proj_cache['u_lp'] = u_new

        return process_activation_target(A_det, A_proj), B_proj, None, None, None, None, None, None, None


class MatMulProjectionHybrid(torch.autograd.Function):
    """
    Hybrid matmul backward: real chain-rule gradients upstream + weight update.

    This decouples two concerns:

      • Upstream signal propagation (∂L/∂A):
            Real chain-rule gradient ``Z_grad @ B^T``.  Flows backward without
            attenuation, so all earlier layers receive a full-magnitude gradient
            signal regardless of network depth.  No vanishing-target problem.

      • Weight update (∂L/∂B):
            When ``use_projections=True``  → returns a **projection target**
                ``B - eta * (A^T @ Z_grad)``  so that ``AlternatingProjections``
                (which does p.copy_(p.grad)) correctly updates the weight.
            When ``use_projections=False`` → returns the **raw gradient**
                ``A^T @ Z_grad``  for Muon/Adam/SGD to process normally. Muon's
                Newton-Schulz step will auto-scale it before updating.

    Args (forward):
        A:          Input activations  (... , M, K)
        B:          Weight matrix      (K, N)
        omega:      Output scale divisor (default 1.0)
        eta:        Step size for gradient→target conversion in projection mode
                    (default 0.01). Ignored when use_projections=False.
        residual:   If True use the residual output ``(A - A @ B) / omega``
    """
    @staticmethod
    def forward(ctx, A, B, omega=1.0, eta=0.01, residual=False, num_steps=1, alpha=1.0, g=1.0, proj_cache=None, p=2.0):
        ctx.save_for_backward(A, B)
        ctx.omega = omega
        ctx.eta = eta
        ctx.residual = residual
        ctx.num_steps = num_steps
        ctx.alpha = alpha
        ctx.g = g
        ctx.proj_cache = proj_cache
        ctx.p = p
        if residual:
            return (A - A @ B) / omega
        return (A @ B) / omega

    @staticmethod
    def backward(ctx, grad_output):
        A, B = ctx.saved_tensors
        A_det = A.detach()
        B_det = B.detach()
        
        # Calculate the mathematical target output Z_target for the projection.
        # Hybrid mode receives real gradients (grad_output), so we construct the target:
        if ctx.residual:
            Z_orig = (A_det - A_det @ B_det) / ctx.omega
        else:
            Z_orig = (A_det @ B_det) / ctx.omega
        Z_det = Z_orig - ctx.eta * grad_output.detach()

        # Scale grad back to the un-normalised space
        Z_grad = grad_output.detach() * ctx.omega

        # Flatten batch dims so matmuls are always 2-D
        if A_det.ndim > 2:
            A_2d = A_det.reshape(-1, A_det.shape[-1])
        else:
            A_2d = A_det
        Z_grad_2d = Z_grad.reshape(-1, Z_grad.shape[-1]) if Z_grad.ndim > 2 else Z_grad

        if ctx.residual:
            IminusB = (torch.eye(B_det.size(0), device=B_det.device, dtype=B_det.dtype) - B_det)
            grad_A = Z_grad @ IminusB.T
        else:
            # ── Real gradient upstream: ∂L/∂A = Z_grad @ B^T  ──────────────
            # Full-magnitude chain-rule gradient — no vanishing across layers.
            grad_A = Z_grad @ B_det.T

        # ── Weight gradient: A^T @ Z_grad  ──────────────────────────────────
        grad_B = A_2d.T @ Z_grad_2d

        if config.use_projections:
            # Projection mode: AlternatingProjections does p.copy_(p.grad),
            # so we must return a TARGET, not a gradient.
            t_init = None
            if ctx.proj_cache is not None:
                t_init = ctx.proj_cache.get('t')

            if A_det.ndim > 2 or B_det.ndim > 2:
                A_2d_proj = A_det.reshape(-1, A_det.shape[-1]).clone()
                Z_2d_proj = Z_det.reshape(-1, Z_det.shape[-1]).clone() * ctx.omega
                B_2d_proj = B_det.reshape(-1, B_det.shape[-2], B_det.shape[-1]).mean(dim=0).clone()
                
                t_init_2d = t_init.reshape(Z_2d_proj.shape) if (t_init is not None and t_init.shape == Z_det.shape) else None
                
                if config.projection_norm in ["inf", "linf", "l_inf", "infinity"]:
                    _, B_proj_2d, _, t_new = matmul_proj_linf(
                        A_2d_proj, B_2d_proj, Z_2d_proj, eps_init=t_init_2d, g=ctx.g, 
                        omega=ctx.omega, num_steps=ctx.num_steps, residual=ctx.residual
                    )
                elif config.projection_norm in ["1", "l1", "l_1", "manhattan"]:
                    _, B_proj_2d, _, t_new = matmul_proj_l1(
                        A_2d_proj, B_2d_proj, Z_2d_proj, eps_init=t_init_2d, g=ctx.g, 
                        omega=ctx.omega, num_steps=ctx.num_steps, residual=ctx.residual
                    )
                elif config.projection_norm in ["lp"] or isinstance(getattr(config, 'projection_p', None), (int, float)):
                    _p = getattr(config, 'projection_p', ctx.p)
                    _, B_proj_2d, _, t_new = matmul_proj_lp(
                        A_2d_proj, B_2d_proj, Z_2d_proj, p=float(_p), u_init=t_init_2d, g=ctx.g, 
                        omega=ctx.omega, num_steps=ctx.num_steps, residual=ctx.residual
                    )
                else:
                    _, B_proj_2d, _, t_new = matmul_proj(
                        A_2d_proj, B_2d_proj, Z_2d_proj, t_init=t_init_2d, alpha=ctx.alpha, g=ctx.g, 
                        omega=ctx.omega, num_steps=ctx.num_steps, residual=ctx.residual
                    )
                
                B_proj = B_proj_2d.reshape(B_det.shape[-2], B_det.shape[-1]).expand(B_det.shape)
                grad_B = B_proj
                if ctx.proj_cache is not None:
                    ctx.proj_cache['t'] = t_new.reshape(Z_det.shape)
            else:
                # .contiguous().clone() ensures no view _base for 2D inputs
                # that may come from transpose() or other view ops
                if config.projection_norm in ["inf", "linf", "l_inf", "infinity"]:
                    A_proj, B_proj, _, t_new = matmul_proj_linf(
                        A_det.contiguous().clone(), B_det.contiguous().clone(), Z_det.contiguous().clone() * ctx.omega,
                        eps_init=t_init, g=ctx.g, omega=ctx.omega, 
                        num_steps=ctx.num_steps, residual=ctx.residual
                    )
                elif config.projection_norm in ["1", "l1", "l_1", "manhattan"]:
                    A_proj, B_proj, _, t_new = matmul_proj_l1(
                        A_det.contiguous().clone(), B_det.contiguous().clone(), Z_det.contiguous().clone() * ctx.omega,
                        eps_init=t_init, g=ctx.g, omega=ctx.omega, 
                        num_steps=ctx.num_steps, residual=ctx.residual
                    )
                elif config.projection_norm in ["lp"] or isinstance(getattr(config, 'projection_p', None), (int, float)):
                    _p = getattr(config, 'projection_p', ctx.p)
                    A_proj, B_proj, _, t_new = matmul_proj_lp(
                        A_det.contiguous().clone(), B_det.contiguous().clone(), Z_det.contiguous().clone() * ctx.omega,
                        p=float(_p), u_init=t_init, g=ctx.g, omega=ctx.omega, 
                        num_steps=ctx.num_steps, residual=ctx.residual
                    )
                else:
                    A_proj, B_proj, _, t_new = matmul_proj(
                        A_det.contiguous().clone(), B_det.contiguous().clone(), Z_det.contiguous().clone() * ctx.omega,
                        t_init=t_init, alpha=ctx.alpha, g=ctx.g, omega=ctx.omega, 
                        num_steps=ctx.num_steps, residual=ctx.residual
                    )
                grad_B = B_proj
                if ctx.proj_cache is not None:
                    ctx.proj_cache['t'] = t_new

        # Reshape grad_A back to original activation shape
        grad_A = grad_A.reshape(A_det.shape)

        # Nones for omega, eta, residual, num_steps, alpha, g, proj_cache, p
        return grad_A, grad_B, None, None, None, None, None, None, None, None


# @torch.compile(dynamic=True)
def orthogonal_rotation_proj(R_0, x_0, y_0, alpha=1.0, gamma=1.0):
    """
    Exact independent projection for R @ x = y, where R is a valid rotation matrix in SO(n).
    
    Minimizes:
        || R - R_0 ||_F^2 + alpha * || x - x_0 ||_2^2 + gamma * || y - y_0 ||_2^2
        subject to R^T R = I, det(R) = 1, and R @ x = y
        
    Handles both single vectors and batched inputs. If batched, returns the shared 
    optimal rotation R* and individual projected vectors x*, y*.
    """
    # Track original dimensionality to handle both (..., N) and (..., N, 1) vectors
    is_1d = False
    if x_0.ndim == R_0.ndim - 1:
        x_0 = x_0.unsqueeze(-1)
        y_0 = y_0.unsqueeze(-1)
        is_1d = True
    elif x_0.ndim >= R_0.ndim and x_0.shape[-1] == R_0.shape[-1]:
        # Handle cases where x_0 is (..., D) but matches R_0 (D, D)
        x_0 = x_0.unsqueeze(-1)
        y_0 = y_0.unsqueeze(-1)
        is_1d = True

    # Weight coefficient for the Procrustes cross term
    denom = alpha + gamma + 1e-8
    w = (alpha * gamma) / denom

    # 1. Construct the Procrustes Matrix (fused operations)
    # If batched, we average over the outer products: M = R_0 + w * avg(y_i @ x_i^T)
    outer = y_0 @ x_0.transpose(-2, -1)
    if outer.ndim > R_0.ndim:
        # Sum/Mean over all dimensions except the last two (which are D, D)
        M_outer = outer.mean(dim=list(range(outer.ndim - 2)))
    else:
        M_outer = outer
        
    M = R_0 + w * M_outer

    # 2. Find the Optimal Rotation (R*) via Batched SVD (or single)
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    
    # Compute determinant to check for reflections (we need pure rotations)
    det_UVh = torch.linalg.det(U @ Vh)
    
    # Create the diagonal correction matrix S_corr to enforce det(R) = 1
    ones = torch.ones_like(S[..., :-1])
    diag = torch.cat([ones, det_UVh.unsqueeze(-1)], dim=-1)
    S_corr = torch.diag_embed(diag)

    # R_proj analytically extracts the closest valid rotation matrix
    R_proj = U @ S_corr @ Vh

    # 3. Find the Optimal Input Vector (x*)
    # Analytically averages the input and inverse-rotated output based on weights
    # Note: R_proj may be broadcasted if x_0/y_0 are batched
    x_proj = (alpha / denom) * x_0 + (gamma / denom) * (R_proj.transpose(-2, -1) @ y_0)

    # 4. Find the Optimal Output Vector (y*)
    y_proj = R_proj @ x_proj

    # Revert dimension expansion if necessary
    if is_1d:
        x_proj = x_proj.squeeze(-1)
        y_proj = y_proj.squeeze(-1)

    return R_proj, x_proj, y_proj


# ─── autograd.Function wrappers ───────────────────────────────────────────────

class OrthogonalRotationProjection(torch.autograd.Function):
    """
    Minimize || R_{new} - R_{old} ||_F^2 + alpha * || x_{new} - x_{old} ||_2^2 + gamma * || y_{new} - y_{old} ||_2^2
    subject to R_{new}^T R_{new} = I, det(R_{new}) = 1, and R_{new} x_{new} = y_{new}.
    """
    @staticmethod
    def forward(ctx, R, x, alpha=1.0, gamma=1.0):
        ctx.save_for_backward(R, x)
        ctx.alpha = alpha
        ctx.gamma = gamma
        
        # Forward pass applies the rotation constraint
        # Use torch.matmul to handle both single vectors and batched inputs
        if x.shape[-1] == R.shape[-1]:
            # Treat x as a vector (or batch of vectors) being rotated
            # (..., D) -> (..., D, 1) then R @ x -> (..., D, 1) then squeeze -> (..., D)
            return torch.matmul(R, x.unsqueeze(-1)).squeeze(-1)
        else:
            return R @ x

    @staticmethod
    def backward(ctx, y_target):
        R, x = ctx.saved_tensors
        
        # Detach and clone contiguously to ensure we break the autograd view 
        # chain properly before pushing to the compiled projection operator
        R_det = R.detach().contiguous().clone()
        x_det = x.detach().contiguous().clone()
        y_det = y_target.detach().contiguous().clone()

        # Execute the exact closed-form geometric projection
        R_proj, x_proj, y_proj = orthogonal_rotation_proj(
            R_det, x_det, y_det, 
            alpha=ctx.alpha, 
            gamma=ctx.gamma
        )

        # Return the modified/projected inputs hijacking standard gradients
        # Nones correspond to the non-tensor hyperparameter inputs (alpha, gamma)
        return R_proj, x_proj, None, None
        
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
        return result
    
class ReLUProjection(torch.autograd.Function):
    """
    Sets the new input directly based on the target z.
    Ignores the original input x entirely.
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
        dist_1 = (x - x_1)**2 + z**2

        # Solution 2: project onto active branch (x > 0, output = x)
        x_2 = torch.clamp((x + z) / 2.0, min=0)
        dist_2 = (x - x_2)**2 + (z - x_2)**2

        # Select solution minimizing distance
        result = torch.where(dist_1 < dist_2, x_1, x_2)
        return result        

class LogitSoftcapInversion(torch.autograd.Function):
    """
    Sets the new input target directly based on the target z.
    Uses the exact mathematical inverse of the softcap function.
    """
    @staticmethod
    def forward(ctx, x, logit_softcap):
        # Save the softcap constant for the backward pass
        ctx.logit_softcap = logit_softcap
        
        # We don't need to save 'x' because pure inversion only relies on 'z'
        return logit_softcap * torch.tanh(x / logit_softcap)

    @staticmethod
    def backward(ctx, z):
        # In Target Propagation, 'z' is the target output. 
        # We must return 'x_target', the input that would produce 'z'.
        C = ctx.logit_softcap
        
        # CRITICAL SAFETY STEP: 
        # arctanh is only valid for inputs strictly between -1 and 1.
        # If the network asks for a target 'z' that is outside the bounds of the softcap,
        # we MUST clamp it slightly inside the bounds to avoid returning NaNs.
        eps = 1e-6
        z_clipped = torch.clamp(z, min=-C + eps, max=C - eps)
        
        # Direct mathematical inversion
        x_target = C * torch.arctanh(z_clipped / C)
        
        # We return x_target for 'x', and None for 'logit_softcap' (as it's a fixed hyperparameter)
        return x_target, None
        
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
        return result, None

class SumReLUProjection(torch.autograd.Function):
    """
    PyTorch equivalent of PJAX sum_relu projection.
    Forward: relu(sum of inputs)
    Backward: projects inputs onto the ReLU constraint graph.
    """
    @staticmethod
    def forward(ctx, *inputs):
        ctx.save_for_backward(*inputs)
        return torch.relu(sum(inputs))

    @staticmethod
    def backward(ctx, z):
        inputs = ctx.saved_tensors
        n = len(inputs)

        # compute intermediate values
        s = sum(inputs)
        mid_1 = s / n
        mid_2 = (z - s) / (n + 1)

        # Solution 1: project onto inactive branch (sum < 0 → output = 0)
        new_inputs_1 = [torch.where(s < 0, x, x - mid_1) for x in inputs]
        dist_1 = torch.where(s < 0, z**2, z**2 + n * mid_1**2)

        # Solution 2: project onto active branch (output = relu(sum))
        new_output_2 = torch.clamp(z - mid_2, min=0)
        new_inputs_2 = [torch.where(new_output_2 == 0, x - mid_1, x + mid_2) for x in inputs]
        dist_2 = torch.where(z == 0, z**2 + n * mid_1**2, (n + 1) * mid_2**2)

        # Select solution minimizing distance
        result = tuple(torch.where(dist_1 < dist_2, x_1, x_2)
                       for x_1, x_2 in zip(new_inputs_1, new_inputs_2))
        return result

def midpoint_softmax_proj_pt(a, z):
    """
    Computes the entropy-regularized projection.
    Finds the probability distribution that is maximally aligned with both 
    the forward logits (a) and the target data (z).
    """
    # Step 1: Find the midpoint of the forward and target data
    m = (a + z) / 2.0
    
    # Step 2: Apply the derived exponential scaling (Softmax)
    projected_p = F.softmax(m, dim=-1)
    
    # Note: Returning as a tuple to match the (projected_a,) style 
    # of your earlier hardmax/simplex projection functions.
    return (projected_p,)


class SoftmaxProjection(torch.autograd.Function):
    """
    PyTorch autograd function using the midpoint Softmax projection 
    """
    @staticmethod
    def forward(ctx, a):
        # Save the raw forward logits 'a' for the backward pass
        ctx.save_for_backward(a)
        
        # The forward pass is just standard softmax
        return F.softmax(a, dim=-1)

    @staticmethod
    def backward(ctx, z_target):
        # Extract the saved forward logits
        (a,) = ctx.saved_tensors
        
        # Apply the mathematically derived midpoint projection
        res = midpoint_softmax_proj_pt(a, z_target)
        
        # Return the projected distribution as the gradient
        return res[0]

class AddProjection(torch.autograd.Function):
    """
    Projects inputs onto the element-wise addition constraint graph.
    Forward: x1 + x2
    Backward: Returns projected x1 and x2 based on target y.
    """
    @staticmethod
    def forward(ctx, x1, x2):
        ctx.save_for_backward(x1, x2)
        return x1 + x2

    @staticmethod
    def backward(ctx, y_target):
        x1, x2 = ctx.saved_tensors
        
        # 1. Compute the residual (delta)
        # How far off is the current forward pass from the target?
        delta = y_target - (x1 + x2)
        
        # 2. Distribute the correction equally across all variables
        # (2 inputs + 1 output = 3 variables), matching pjax's sum_proj:
        #   t = (z - a.sum()) / (a.size + z.size)
        correction = delta / 3.0
        
        # 3. Compute projected inputs
        x1_star = x1 + correction
        x2_star = x2 + correction
        
        return x1_star, x2_star

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
def simplex_op_pt(a):
    """Project onto probability simplex (forward operation). Batched version."""
    n = a.size(-1)
    
    # Sort array descending
    u, _ = torch.sort(a, dim=-1, descending=True)
    
    # Compute cumulative sums to calculate candidates for the shift tau
    cssv = torch.cumsum(u, dim=-1)
    k_range = torch.arange(1, n + 1, device=a.device, dtype=a.dtype)
    tau_candidates = (cssv - 1.0) / k_range
    
    # Valid active sets are those where the element is greater than the shift.
    valid_mask = u > tau_candidates
    rho = torch.sum(valid_mask, dim=-1, keepdim=True)
    rho = torch.clamp(rho, min=1)
    
    # Extract the correct tau
    tau = torch.gather(tau_candidates, -1, rho - 1)
    
    return torch.maximum(a - tau, torch.tensor(0.0, device=a.device, dtype=a.dtype))

def simplex_proj_pt(a, z):
    """Project onto the probability simplex function graph. Batched version."""
    n = a.size(-1)

    # Combine inputs to find the optimal active sets based on the "pull"
    w = a + z
    
    # Get indices to sort ascending, keeping track to restore order later
    idx = torch.argsort(w, dim=-1)
    a_s = torch.gather(a, -1, idx)
    z_s = torch.gather(z, -1, idx)
    w_s = torch.gather(w, -1, idx)
    
    # Work in descending order (index 0 is the largest combined pull)
    a_d = torch.flip(a_s, dims=[-1])
    z_d = torch.flip(z_s, dims=[-1])
    w_d = torch.flip(w_s, dims=[-1])
    
    # Precompute tau for all candidates k=1..n
    k_range = torch.arange(1, n + 1, device=a.device, dtype=a.dtype)
    tau = (torch.cumsum(w_d, dim=-1) - 2.0) / k_range
    tau_matrix = tau.unsqueeze(-1)
    
    # Mask for active set: lower triangular matrix of True
    # Row k (0-indexed) corresponds to candidate k+1, having k+1 active elements
    active_mask = torch.tril(torch.ones((n, n), dtype=torch.bool, device=a.device))
    
    # Calculate candidate arrays
    # 1. Active elements
    w_d_row = w_d.unsqueeze(-2)
    a_active = (w_d_row + tau_matrix) / 2.0
    z_active = (w_d_row - tau_matrix) / 2.0
    
    # 2. Clamped elements
    a_d_row = a_d.unsqueeze(-2)
    a_clamped = torch.minimum(a_d_row, tau_matrix)
    z_clamped = torch.zeros_like(z_active)
    
    # Combine based on the active mask
    a_cand = torch.where(active_mask, a_active, a_clamped)
    z_cand = torch.where(active_mask, z_active, z_clamped)
    
    z_d_row = z_d.unsqueeze(-2)
    
    # Calculate squared euclidean distances
    dist = torch.sum((a_cand - a_d_row)**2 + (z_cand - z_d_row)**2, dim=-1)
    
    # Select valid candidates
    # A candidate is valid if its smallest active w element (which is exactly w_d[k]) is >= tau[k]
    dist_valid = torch.where(w_d >= tau, dist, torch.tensor(float('inf'), device=a.device, dtype=a.dtype))
    
    # Select the candidate minimizing the distance
    best_k = torch.argmin(dist_valid, dim=-1, keepdim=True)
    
    # We must extract the single best candidate array out of 'n' potential candidates along dim=-2
    best_k_expanded = best_k.unsqueeze(-1).expand(*best_k.shape[:-1], 1, n)
    best_a_d = torch.gather(a_cand, -2, best_k_expanded).squeeze(-2)
    
    # Restore the original order: flip back to ascending, then apply inverse argsort
    best_a_s = torch.flip(best_a_d, dims=[-1])
    
    inverse_idx = torch.argsort(idx, dim=-1)
    best_a_orig = torch.gather(best_a_s, -1, inverse_idx)
    return (best_a_orig,)


def hardmax_op_pt(a):
    """Project onto the hardmax vertices (forward operation)."""
    winner = torch.argmax(a, dim=-1, keepdim=True)
    return torch.zeros_like(a).scatter_(-1, winner, 1.0)


def _project_hardmax_winner_region_pt(a, winner):
    """Project rows of ``a`` onto the region where ``winner`` is an argmax."""
    n = a.size(-1)
    if n == 1:
        return a.clone()

    winner = winner.to(dtype=torch.long)
    winner_value = torch.gather(a, -1, winner)

    winner_mask = F.one_hot(winner.squeeze(-1), num_classes=n).to(dtype=torch.bool, device=a.device)
    others = a.masked_fill(winner_mask, float("-inf"))
    sorted_others, _ = torch.sort(others, dim=-1, descending=True)
    sorted_others = sorted_others[..., :-1]

    prefix = torch.cumsum(sorted_others, dim=-1)
    counts = torch.arange(2, n + 1, device=a.device, dtype=a.dtype)

    candidate_t = torch.empty_like(a)
    candidate_t[..., :1] = winner_value
    candidate_t[..., 1:] = (winner_value + prefix) / counts

    lower_bounds = torch.empty_like(candidate_t)
    lower_bounds[..., :-1] = sorted_others
    lower_bounds[..., -1] = float("-inf")
    valid = candidate_t >= lower_bounds

    candidate_index = valid.to(torch.int64).argmax(dim=-1, keepdim=True)
    threshold = torch.gather(candidate_t, -1, candidate_index)

    projected = torch.minimum(a, threshold)
    projected.scatter_(-1, winner, threshold)
    return projected


def hardmax_proj_pt(a, z):
    """Approximate projection onto the hardmax function graph.

    The output branch is selected by the closest one-hot vertex to ``z``
    (equivalently ``argmax(z)``), and ``a`` is projected onto the region where
    that branch is a valid hardmax winner.
    """
    winner = torch.argmax(z, dim=-1, keepdim=True)
    projected_a = _project_hardmax_winner_region_pt(a, winner)
    return (projected_a,)

class SimplexProjection(torch.autograd.Function):
    """
    PyTorch equivalent of PJAX simplex projection.
    Forward: projects a onto the probability simplex.
    Backward: projects inputs onto the simplex constraint graph.
    """
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return simplex_op_pt(a)

    @staticmethod
    def backward(ctx, z_target):
        (a,) = ctx.saved_tensors
        res = simplex_proj_pt(a, z_target)
        return res[0]


class HardmaxProjection(torch.autograd.Function):
    """
    Projection onto hardmax vertices.

    Forward: returns the one-hot hardmax of the input.
    Backward: projects the pre-activation onto the winner region selected by
    the downstream target's closest one-hot vertex.
    """
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return hardmax_op_pt(a)

    @staticmethod
    def backward(ctx, z_target):
        (a,) = ctx.saved_tensors
        res = hardmax_proj_pt(a, z_target)
        return res[0]


class LayerNormProjection(torch.autograd.Function):
    """
    Projects inputs onto a locally linearized LayerNorm constraint graph.
    Assumes standard LayerNorm without learned gamma/beta parameters.
    """
    @staticmethod
    def forward(ctx, x, eps=1e-5):
        # 1. Compute standard forward LayerNorm
        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        sigma = torch.sqrt(var + eps)
        
        y_fwd = (x - mu) / sigma
        
        # 2. Save forward statistics and output for the projection
        ctx.save_for_backward(mu, sigma, y_fwd)
        
        return y_fwd

    @staticmethod
    def backward(ctx, y_target):
        mu_0, sigma_0, y_fwd = ctx.saved_tensors
        
        # 1. Calculate projected output (y*)
        # We use the variance (sigma_0^2) for the weighted average
        var_0 = sigma_0 ** 2
        y_star = (y_target + var_0 * y_fwd) / (var_0 + 1.0)
        
        # 2. Calculate projected input (x*) using the fixed affine relationship
        x_star = sigma_0 * y_star + mu_0
        
        # Return projected input
        # Note: Return None for 'eps' as it requires no gradient/projection
        return x_star, None


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


# @torch.compile(dynamic=True)
def exact_batchnorm_proj(x, z, eps=1e-5, num_steps=15):
    """
    Exact projection onto the non-linear BatchNorm constraint graph.
    Solves the coupled quartic polynomial for the optimal standard deviation 
    using pointwise Newton's method in O(N) time.
    """
    # 1. Compute centered statistics over the batch dimension (dim=0)
    mu_x = x.mean(dim=0, keepdim=True)
    mu_z = z.mean(dim=0, keepdim=True)

    x_c = x - mu_x
    z_c = z - mu_z

    v_x = (x_c.square()).mean(dim=0, keepdim=True)
    v_z = (z_c.square()).mean(dim=0, keepdim=True)
    cov = (x_c * z_c).mean(dim=0, keepdim=True)

    # 2. Initialize 'a' (the optimal standard deviation scale for x*)
    # The standard deviation of the input x is a mathematically ideal starting point
    a = torch.sqrt(v_x + eps)

    # 3. Newton's Method to find the root of the quartic derivative
    for _ in range(num_steps):
        # S is the norm of the combined target vector
        S = torch.sqrt(a.square() * v_x + 2.0 * a * cov + v_z + eps)
        
        # Evaluate function g(a) = 0
        g = a * S - a * v_x - cov
        
        # Evaluate derivative g'(a)
        S_inv = 1.0 / (S + eps)
        g_prime = S + a * (a * v_x + cov) * S_inv - v_x
        
        # Update step (clamped to ensure variance stays non-negative)
        step = g / (g_prime + 1e-8)
        a = torch.relu(a - step)

    # 4. Construct the exact projected variables
    S_final = torch.sqrt(a.square() * v_x + 2.0 * a * cov + v_z + eps)
    
    # Projected y* is guaranteed to have exactly zero mean and unit variance
    y_star = (a * x_c + z_c) / (S_final + eps)
    
    # Projected x* is the exact affine shift of y*
    x_star = a * y_star + mu_x
    
    return x_star, y_star

# @torch.compile(dynamic=True)
def exact_batchnorm_proj(x, z, eps=1e-5, num_steps=5):
    """
    Exact pointwise Newton projection onto the non-linear BatchNorm constraint graph.
    Returns both the unnormalized x_star and the normalized x_hat_star.
    """
    mu_x = x.mean(dim=0, keepdim=True)
    mu_z = z.mean(dim=0, keepdim=True)

    x_c = x - mu_x
    z_c = z - mu_z

    v_x = (x_c.square()).mean(dim=0, keepdim=True)
    v_z = (z_c.square()).mean(dim=0, keepdim=True)
    cov = (x_c * z_c).mean(dim=0, keepdim=True)

    a = torch.sqrt(v_x + eps)

    for _ in range(num_steps):
        S = torch.sqrt(a.square() * v_x + 2.0 * a * cov + v_z + eps)
        g = a * S - a * v_x - cov
        
        S_inv = 1.0 / (S + eps)
        g_prime = S + a * (a * v_x + cov) * S_inv - v_x
        
        step = g / (g_prime + 1e-8)
        a = torch.relu(a - step)

    S_final = torch.sqrt(a.square() * v_x + 2.0 * a * cov + v_z + eps)
    
    # x_hat_star is the strictly normalized constraint (mean=0, var=1)
    x_hat_star = (a * x_c + z_c) / (S_final + eps)
    
    # x_star is the pre-normalized input
    x_star = a * x_hat_star + mu_x
    
    return x_star, x_hat_star


class AffineBatchNormProjection(torch.autograd.Function):
    """
    Minimizes || x' - x ||^2 + || gamma' - gamma ||^2 + || beta' - beta ||^2 + || y' - y ||^2
    Subject to: y' = gamma' * BN(x') + beta'
    """
    @staticmethod
    def forward(ctx, x, weight, bias, eps=1e-5, num_steps=3):
        mu = x.mean(dim=0, keepdim=True)
        var = x.var(dim=0, keepdim=True, unbiased=False)
        
        # Save the normalized x for the forward pass
        x_hat = (x - mu) / torch.sqrt(var + eps)
        y = weight * x_hat + bias
        
        ctx.save_for_backward(x, weight, bias)
        ctx.eps = eps
        ctx.num_steps = num_steps
        
        return y

    @staticmethod
    def backward(ctx, z_target):
        x, weight, bias = ctx.saved_tensors
        eps = ctx.eps
        N = x.size(0)
        
        # 1. Map target back to the pre-affine space to project X
        # We invert the scale and shift: z_shifted = (z - beta) / gamma
        z_shifted = (z_target - bias) / (weight + 1e-8)
        
        # 2. Exact projection for X (getting both the raw and normalized updates)
        x_star, x_hat_star = exact_batchnorm_proj(x, z_shifted, eps=eps, num_steps=ctx.num_steps)
        
        # 3. Exact analytical projection for gamma (weight) and beta (bias)
        # Because x_hat_star is guaranteed to have mean=0 and var=1, 
        # the optimal updates for gamma and beta decouple cleanly from each other.
        beta_star = (bias + z_target.sum(dim=0)) / (1.0 + N)
        weight_star = (weight + (x_hat_star * z_target).sum(dim=0)) / (1.0 + N)
        
        # Return projected inputs. None for kwargs.
        return x_star, weight_star, beta_star, None, None


# @torch.compile(dynamic=True)
def exact_rmsnorm_proj(x, z, eps=1e-5, num_steps=5):
    """
    Exact pointwise Newton projection onto the RMSNorm constraint graph.

    Constraint: y = x / RMS(x),  where RMS(x) = sqrt(mean(x^2) + eps).

    Given (x0, z0), finds (x*, y*) minimizing ||x* - x0||^2 + ||y* - z0||^2
    subject to y* = x* / RMS(x*).

    Parameterization: x* = a * u,  y* = u * sqrt(n) / len_u  where u is the
    optimal direction and a = RMS(x*). The problem reduces to a 1D root-find
    for the scalar 'a' per normalization group.

    Returns both the pre-normalized x_star and the normalized y_star.
    """
    # Statistics over the feature dimension (last dim)
    v_x = (x.square()).mean(dim=-1, keepdim=True)
    v_z = (z.square()).mean(dim=-1, keepdim=True)
    cov = (x * z).mean(dim=-1, keepdim=True)

    # Initialize 'a' at the current RMS of x
    a = torch.sqrt(v_x + eps)

    # Newton's method: solve g(a) = a*S - a*v_x - cov = 0
    for _ in range(num_steps):
        S = torch.sqrt(a.square() * v_x + 2.0 * a * cov + v_z + eps)

        g = a * S - a * v_x - cov

        S_inv = 1.0 / (S + eps)
        g_prime = S + a * (a * v_x + cov) * S_inv - v_x

        step = g / (g_prime + 1e-8)
        a = torch.relu(a - step)

    # Reconstruct the projected variables
    S_final = torch.sqrt(a.square() * v_x + 2.0 * a * cov + v_z + eps)

    # y_star: the RMS-normalized output (has unit RMS by construction)
    y_star = (a * x + z) / (S_final + eps)

    # x_star: the pre-normalized input
    x_star = a * y_star

    return x_star, y_star


class RMSNormProjection(torch.autograd.Function):
    """
    Projection onto the RMSNorm constraint graph.

    Forward:  y = x / RMS(x)            (no gain)
              y = weight * x / RMS(x)   (with gain)

    Backward: exact Newton projection onto the nonlinear RMS constraint,
              with gain handled analytically in the pre-gain space.
    """
    @staticmethod
    def forward(ctx, x, weight=None, eps=1e-5, num_steps=5):
        rms = torch.sqrt(x.square().mean(dim=-1, keepdim=True) + eps)
        x_hat = x / rms

        if weight is not None:
            y = weight * x_hat
        else:
            y = x_hat

        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        ctx.num_steps = num_steps

        return y

    @staticmethod
    def backward(ctx, z_target):
        x, weight = ctx.saved_tensors
        eps = ctx.eps

        if weight is not None:
            # Map target back to pre-gain space: z_shifted = z_target / weight
            z_shifted = z_target / (weight + 1e-8)
        else:
            z_shifted = z_target

        # Exact Newton projection for x
        x_star, x_hat_star = exact_rmsnorm_proj(
            x, z_shifted, eps=eps, num_steps=ctx.num_steps
        )

        if weight is not None:
            # Analytical projection for gain:
            # x_hat_star has unit RMS, so <x_hat_star, z_target> gives the
            # optimal weight update direction per feature.
            n = x.size(0)  # batch size for averaging proposals
            weight_star = (weight + (x_hat_star * z_target).sum(dim=0)) / (1.0 + n)
            return x_star, weight_star, None, None
            
class SquaredReLUProjection(torch.autograd.Function):
    """
    Exact Euclidean projection onto the Squared ReLU graph y = max(0, x)^2.
    Uses fused Newton's method to resolve depressed cubic equation for the active branch.
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.square(torch.relu(x))

    @staticmethod
    def backward(ctx, z_target):
        x, = ctx.saved_tensors
        
        # Branch 1: Inactive (x* <= 0) => y* = 0
        x1 = torch.clamp(x, max=0)
        dist1_sq = (x - x1)**2 + z_target**2

        # Branch 2: Active (x* > 0) => y* = x*^2
        # Solve Depressed Cubic: 2t^3 + (1 - 2*z_target)*t - x = 0
        t = torch.clamp((x + z_target) / 2.0, min=1e-5)
        for _ in range(5):
            f = 2.0 * t**3 + (1.0 - 2.0 * z_target) * t - x
            f_prime = 6.0 * t**2 + (1.0 - 2.0 * z_target)
            step = f / (f_prime.abs() + 1e-6)
            t = torch.relu(t - step) + 1e-6

        x2 = t
        y2 = x2**2
        dist2_sq = (x - x2)**2 + (y2 - z_target)**2

        # Select branch minimizing the distance
        x_star = torch.where(dist1_sq < dist2_sq, x1, x2)
        
        return x_star