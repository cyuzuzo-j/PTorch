import torch
import torch.nn.functional as F
import warnings
from ..config import config
import math
from itertools import repeat


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
    
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
        
    hs = _POLAR_COEFFS[:steps]
    if steps > len(_POLAR_COEFFS):
        hs += list(repeat(_POLAR_COEFFS[-1], steps - len(_POLAR_COEFFS)))
        
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
    if not config.use_muon_activations:
        return A_proj

    orig_shape = A_det.shape
    last_dim = orig_shape[-1]

    if last_dim == 0 or A_det.numel() == 0:
        return A_proj

    A_2d = A_det.reshape(-1, last_dim)
    A_proj_2d = A_proj.reshape(-1, last_dim)
    g = A_2d - A_proj_2d

    g_muon = zeropower_via_polarexpress(g)

    A_proj_new = A_2d - config.muon_activations_lr * g_muon

    return A_proj_new.reshape(orig_shape)



        

@torch.compile()
def matmul_proj_linf(A, B, Z, eps_init=None, g=1.0, omega=1.0, num_steps=5):
    """
    Exact independent bilinear projection for A @ B = Z using the L_infinity (Chebyshev) norm.    
    """
    M = A.size(-2)
    N = B.size(-1)

    # 1. Base pairwise projection
    P = A @ B  # (..., M, N)
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
    B_ext = B.unsqueeze(-3)   # (..., 1, K, N)
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
    B_proj = B + dB.mean(dim=-3)
    
    Z_proj = Z - S * (eps / (g * omega))
            
    return A_proj, B_proj, Z_proj, eps.detach()


class MatMulProjectionLinf(torch.autograd.Function):
    """
    Minimize || A_{new} - A_{old} ||_inf + || B_{new} - B_{old} ||_inf + g * || Z_{new} - Z_{old} ||_inf
    subject to A_{new} @ B_{new} = Z_{new} (or A_{new} - A_{new} @ B_{new} = Z_{new} if residual)
    """
    @staticmethod
    def forward(ctx, A, B, num_steps, g, omega, proj_cache=None):
        ctx.save_for_backward(A, B)
        ctx.g = g
        ctx.num_steps = num_steps
        ctx.proj_cache = proj_cache  
        ctx.omega = omega
        return (A @ B) / omega

    @staticmethod
    def backward(ctx, Z_target):
        A, B = ctx.saved_tensors
        A_det = A.detach()
        B_det = B.detach()
        Z_det = Z_target.detach()

        eps_init = None
        if ctx.proj_cache is not None:
            eps_init = ctx.proj_cache.get('eps')

        if (A_det.ndim > 2 or B_det.ndim > 2):
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
                num_steps=ctx.num_steps
            )

            if ctx.proj_cache is not None:
                ctx.proj_cache['eps'] = eps_new

        return process_activation_target(A_det, A_proj), B_proj, None, None, None, None, None, None        

@torch.compile()
def matmul_proj(A, B, Z, t_init=None, alpha=1.0, g=1.0, omega=1.0, num_steps=1, residual=False):
    """
    Exact bilinear projection for A @ B = Z (or A - A @ B = Z if residual=True).
    """
    M = A.size(-2)
    N = B.size(-1)


    # 1. Compute pairwise operations in O(M*N) without 3D expansion
    p = A @ B # (..., M, N)
    qa = (A * A).sum(dim=-1, keepdim=True)  # (..., M, 1)
    qb = (B * B).sum(dim=-2, keepdim=True)  # (..., 1, N)
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
    A_proj = (alpha / N) * (A * sum_inv_denom_j + t_inv_denom @ B.transpose(-2, -1))

    # B_proj analytically averages over M proposals via Matrix Math
    sum_inv_denom_i = inv_denom.sum(dim=-2, keepdim=True) # (..., 1, N)
    B_proj = (1.0 / M) * (alpha * B * sum_inv_denom_i + A.transpose(-2, -1) @ t_inv_denom)

    Z_proj = Z - t * target_penalty
    
    return A_proj, B_proj, Z_proj, t.detach()

# ─── autograd.Function wrappers ───────────────────────────────────────────────
class MatMulProjection(torch.autograd.Function):
    """
    Minimize || A_{new} - A_{old} ||_F^2 + alpha * || B_{new} - B_{old} ||_F^2 + g * || Z_{new} - Z_{old} ||_F^2
    subject to A_{new} @ B_{new} = Z_{new} (or A_{new} - A_{new} @ B_{new} = Z_{new} if residual)
    """
    @staticmethod
    def forward(ctx, A, B, num_steps, alpha, g, omega, proj_cache=None, pairwise=False, forward_cache=None):
        ctx.save_for_backward(A, B)
        ctx.alpha = alpha
        ctx.g = g
        ctx.num_steps = num_steps
        ctx.proj_cache = proj_cache  
        ctx.forward_cache = forward_cache 
        ctx.omega = omega
        ctx.pairwise = pairwise
        return (A @ B) / omega

    @staticmethod
    def backward(ctx, Z_target):
        A, B = ctx.saved_tensors
        A_det = A.detach()
        B_det = B.detach()
        Z_det = Z_target.detach()

        t_init = None
        if not ctx.pairwise and ctx.proj_cache is not None:
            t_init = ctx.proj_cache.get('t')

        if not ctx.pairwise and (A_det.ndim > 2 or B_det.ndim > 2):
            A_2d = A_det.reshape(-1, A_det.shape[-1]).clone()
            Z_2d = Z_det.reshape(-1, Z_det.shape[-1]).clone() * ctx.omega
            B_2d = B_det.reshape(-1, B_det.shape[-2], B_det.shape[-1]).mean(dim=0).clone()
            
            t_init_2d = t_init.reshape(Z_2d.shape) if (t_init is not None and t_init.shape == Z_det.shape) else None
            
            A_proj_2d, B_proj_2d, Z_proj_2d, t_new = matmul_proj(
                A_2d, B_2d, Z_2d, t_init=t_init_2d, alpha=ctx.alpha, g=ctx.g, 
                omega=ctx.omega, num_steps=ctx.num_steps
            )
            
            A_proj = A_proj_2d.reshape(A_det.shape)
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
                num_steps=ctx.num_steps
            )

            # Update the cache for the next iteration if not pairwise
            if not ctx.pairwise and ctx.proj_cache is not None:
                ctx.proj_cache['t'] = t_new

            B_proj = B_proj

        if ctx.forward_cache is not None:
            ctx.forward_cache[0] = Z_proj
        return process_activation_target(A_det, A_proj), B_proj, None, None, None, None, None, None, None, None


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
        
        new_logits[cond_0] = torch.maximum(logits[cond_0] - lmbda, torch.zeros_like(logits[cond_0]))
        
        # Move towards the label by at most lambda, but don't overshoot the label
        new_logits[cond_1] = torch.minimum(logits[cond_1] + lmbda, labels[cond_1])

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
        return grad

class ReLULInfinityProjection(torch.autograd.Function):
    """
    Sets the new input directly based on the target z,
    projecting onto the ReLU graph using the L-infinity norm.
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
        # L-infinity distance: max(|x - x_1|, |z - 0|)
        dist_1 = torch.max(torch.abs(x - x_1), torch.abs(z))

        # Solution 2: project onto active branch (x > 0, output = x)
        x_2 = torch.clamp((x + z) / 2.0, min=0)
        # L-infinity distance: max(|x - x_2|, |z - x_2|)
        dist_2 = torch.max(torch.abs(x - x_2), torch.abs(z - x_2))

        # Select solution minimizing L-infinity distance
        result = torch.where(dist_1 < dist_2, x_1, x_2)
        
        result_forwards = torch.where(dist_1 < dist_2, torch.zeros_like(x), x_2)
        ctx.forward_cache[0] = result_forwards        

        return process_activation_target(x, result), None
    
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


def max_proj_pt_batch(a, z):
    """
    Vectorized projection onto the maximum function graph.
    Processes a batch of arrays simultaneously.
    
    Args:
        a: Tensor of shape (B, P) where P is the patch size (e.g., kernel_h * kernel_w).
        z: Tensor of shape (B, 1) containing the target maximums.
    Returns:
        Projected tensor of shape (B, P).
    """
    B, P = a.shape
    
    # 1. Sort arrays
    a_sorted, idx = torch.sort(a, dim=1)

    # 2. Compute candidate maxima (z_k)
    a_sorted_flipped = torch.flip(a_sorted, dims=[1])
    cumsum_flipped = torch.cumsum(a_sorted_flipped, dim=1)
    divisors = torch.arange(2, P + 2, device=a.device, dtype=a.dtype).unsqueeze(0)
    
    z_k_flipped = (cumsum_flipped + z) / divisors
    z_k = torch.flip(z_k_flipped, dims=[1])

    # 3. Compute candidate arrays (a_k)
    # Create upper triangular mask (1, P, P)
    i_ge_k = torch.triu(torch.ones((P, P), dtype=torch.bool, device=a.device)).unsqueeze(0)
    
    # a_k shape: (B, P_candidate, P_element)
    a_k = torch.where(i_ge_k, z_k.unsqueeze(2), a_sorted.unsqueeze(1))

    # 4. Compute distances
    dist = ((a_k - a_sorted.unsqueeze(1)) ** 2).sum(dim=2) + (z_k - z) ** 2

    # 5. Select valid candidates
    # Add a small epsilon (1e-5) to handle floating point inaccuracies
    valid = torch.max(a_k, dim=2)[0] <= z_k + 1e-5
    dist_valid = torch.where(valid, dist, torch.tensor(float('inf'), device=a.device, dtype=dist.dtype))

    # 6. Select candidate minimizing distance
    k = torch.argmin(dist_valid, dim=1)

    # Extract the best sorted candidate array for each batch item
    batch_indices = torch.arange(B, device=a.device)
    best_a_k_sorted = a_k[batch_indices, k, :]

    # 7. Unsort back to original spatial layout
    a_proj = torch.zeros_like(a)
    a_proj.scatter_(1, idx, best_a_k_sorted)

    return a_proj


class MaxPool2DProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, kernel_size, stride=None, padding=0):
        if stride is None:
            stride = kernel_size
        
        k_tuple = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        s_tuple = (stride, stride) if isinstance(stride, int) else tuple(stride)
        
        if k_tuple != s_tuple:
            raise ValueError("MaxPool projection requires strides == pool_size")

        ctx.save_for_backward(input)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        
        return F.max_pool2d(input, kernel_size, stride, padding)

    @staticmethod
    def backward(ctx, z_target):
        input, = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        padding = ctx.padding

        N, C, H_in, W_in = input.shape
        _, _, H_out, W_out = z_target.shape

        kH = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
        kW = kernel_size[1] if isinstance(kernel_size, tuple) else kernel_size

        # 1. Extract patches using unfold
        # Shape: (N, C * kH * kW, H_out * W_out)
        patches = F.unfold(input, kernel_size, stride=stride, padding=padding)
        L = patches.shape[-1] # Number of spatial patches (H_out * W_out)
        
        # 2. Reshape to isolate each local pooling window
        # (N, C, kH * kW, L) -> Permute to (N, C, L, kH * kW)
        patches = patches.view(N, C, kH * kW, L).permute(0, 1, 3, 2).contiguous()
        
        # Flatten batch, channels, and spatial dims into a single batch dimension
        a_batch = patches.view(-1, kH * kW) # Shape: (B, P)
        z_batch = z_target.reshape(-1, 1)      # Shape: (B, 1)

        # 3. Apply vectorized projection
        a_proj_batch = max_proj_pt_batch(a_batch, z_batch)

        # 4. Reconstruct the image
        a_proj_patches = a_proj_batch.view(N, C, L, kH * kW)
        
        a_proj_unfolded = a_proj_patches.permute(0, 1, 3, 2).contiguous().view(N, C * kH * kW, L)

        a_proj = F.fold(
            a_proj_unfolded, 
            output_size=(H_in, W_in), 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding
        )

        
        return a_proj, None, None, None


def extract_patches(input, kernel_size, stride, padding) -> torch.Tensor:
    """Unfolds inputs into spatial patches and permutes for dense layers."""
    patches = F.unfold(input, kernel_size, dilation=1, padding=padding, stride=stride)
    
    H_out = (input.shape[2] + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
    W_out = (input.shape[3] + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
    
    # Reshape to (N, C*kH*kW, H_out, W_out) then permute to (N, H_out, W_out, C*kH*kW)
    return patches.view(input.shape[0], -1, H_out, W_out).permute(0, 2, 3, 1)


class ConvPatchProjection(torch.autograd.Function):
    """
    Handles the Spatial Consensus for Convolutional Activations.
    """
    @staticmethod
    def forward(ctx, input, kernel_size, stride, padding):
        ctx.save_for_backward(input)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        
        patches = F.unfold(input, kernel_size, dilation=1, padding=padding, stride=stride)
        
        kH = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
        kW = kernel_size[1] if isinstance(kernel_size, tuple) else kernel_size
        sH = stride[0] if isinstance(stride, tuple) else stride
        sW = stride[1] if isinstance(stride, tuple) else stride
        pad_h = padding[0] if isinstance(padding, tuple) else padding
        pad_w = padding[1] if isinstance(padding, tuple) else padding
        
        H_out = (input.shape[2] + 2 * pad_h - kH) // sH + 1
        W_out = (input.shape[3] + 2 * pad_w - kW) // sW + 1
        ctx.H_out = H_out
        ctx.W_out = W_out
        
        return patches.view(input.shape[0], -1, H_out, W_out).permute(0, 2, 3, 1)

    @staticmethod
    def backward(ctx, z_target):
        input, = ctx.saved_tensors
        N, C, H, W = input.shape
        
        kH = ctx.kernel_size[0] if isinstance(ctx.kernel_size, tuple) else ctx.kernel_size
        kW = ctx.kernel_size[1] if isinstance(ctx.kernel_size, tuple) else ctx.kernel_size
        
        L = ctx.H_out * ctx.W_out
        
        z_patches = z_target.permute(0, 3, 1, 2).reshape(N, -1, L)
        
        target_sum = F.fold(
            z_patches, 
            output_size=(H, W), 
            kernel_size=ctx.kernel_size, 
            padding=ctx.padding, 
            stride=ctx.stride
        )
        
        dummy_ones = torch.ones(1, kH * kW, L, device=input.device, dtype=input.dtype)
        overlap_counts = F.fold(
            dummy_ones, 
            output_size=(H, W), 
            kernel_size=ctx.kernel_size, 
            padding=ctx.padding, 
            stride=ctx.stride
        )
        
        target_img = target_sum / torch.clamp(overlap_counts, min=1.0)
        
        target = process_activation_target(input, target_img)
        
        return target, None, None, None


