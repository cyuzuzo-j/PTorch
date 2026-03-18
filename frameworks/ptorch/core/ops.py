import torch
import torch.nn.functional as F
import warnings
from .. import config
import math
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
        

@torch.compile(dynamic=True)
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
        return A_proj, B_proj, None, None, None, None, None, None        
@torch.compile(dynamic=True)
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
        return A_proj, B_proj, None, None, None, None, None, None, None


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


@torch.compile(dynamic=True)
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

@torch.compile(dynamic=True)
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