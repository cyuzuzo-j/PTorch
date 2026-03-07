import torch
import torch.nn.functional as F
import warnings
from .. import config

# Suppress pin_memory warning when no accelerator is available
warnings.filterwarnings("ignore", message=".*pin_memory.*")

@torch.compile
def matmul_proj(A, B, Z, t_init=None, alpha=1.0, g=1.0, omega=1.0, num_steps=1):
    """
    Exact independent bilinear projection for A @ B = Z.
    
    Analytically optimized to avoid O(M*N*K) memory expansions. 
    Pointwise Newton method runs in O(M*N) and consensus averaging 
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Shape mismatch: A is ({M},{K}), B is ({K},{N})"

    # 1. Compute pairwise operations in O(M*N) without 3D expansion
    p = A @ B  # (M, N)
    qa = (A * A).sum(dim=1, keepdim=True)  # (M, 1)
    qb = (B * B).sum(dim=0, keepdim=True)  # (1, N)
    q_eff = qa + alpha * qb  # (M, N)

    # 2. Caching logic for t
    if t_init is not None and t_init.shape == p.shape:
        t = t_init.to(p.device)
    else:
        t = torch.zeros_like(p)

    max_t = 0.999 * (alpha ** 0.5)
    target_penalty = (omega ** 2) / (g ** 2)

    # 3. Newton's Method (fused pointwise ops)
    for _ in range(num_steps):
        t2 = t.square()
        alpha_minus_t2 = alpha - t2

        N_num = alpha * (p * (alpha + t2) + t * q_eff)
        f_val = (N_num / (alpha_minus_t2.square())) - Z + t * target_penalty

        N_prime = alpha * (2.0 * t * p + q_eff)
        f_prime_val = ((N_prime * alpha_minus_t2) + 4.0 * t * N_num) / (alpha_minus_t2 ** 3) + target_penalty

        step = f_val / (f_prime_val + 1e-8)
        t = t - step

    # 4. Analytical Consensus Reconstruction
    t2 = t.square()
    denom = alpha - t2
    inv_denom = 1.0 / denom       # (M, N)
    t_inv_denom = t / denom       # (M, N)

    # A_proj analytically averages over N proposals via Matrix Math
    sum_inv_denom_j = inv_denom.sum(dim=1, keepdim=True) # (M, 1)
    A_proj = (alpha / N) * (A * sum_inv_denom_j + t_inv_denom @ B.t())

    # B_proj analytically averages over M proposals via Matrix Math
    sum_inv_denom_i = inv_denom.sum(dim=0, keepdim=True) # (1, N)
    B_proj = (1.0 / M) * (alpha * B * sum_inv_denom_i + A.t() @ t_inv_denom)

    Z_proj = Z - t * target_penalty

    return A_proj, B_proj, Z_proj, t.detach()

# ─── autograd.Function wrappers ───────────────────────────────────────────────
class MatMulProjection(torch.autograd.Function):
    """
    Minimize || A_{new} - A_{old} ||_F^2 + alpha * || B_{new} - B_{old} ||_F^2 + g * || Z_{new} - Z_{old} ||_F^2
    subject to A_{new} @ B_{new} = Z_{new}
    """
    @staticmethod
    def forward(ctx, A, B, proj_cache=None, alpha=1.0, g=1.0, omega=1.0, num_steps=1):
        ctx.save_for_backward(A, B)
        ctx.alpha = alpha
        ctx.g = g
        ctx.num_steps = num_steps
        ctx.proj_cache = proj_cache  # Store reference to the mutable dictionary
        ctx.omega = omega
        return A @ B

    @staticmethod
    def backward(ctx, Z_target):
        A, B = ctx.saved_tensors
        A_det = A.detach()
        B_det = B.detach()
        Z_det = Z_target.detach()

        # Retrieve the cached t from previous runs if available
        t_init = ctx.proj_cache.get('t') if ctx.proj_cache is not None else None

        if A_det.ndim > 2 or B_det.ndim > 2:
            A_2d = A_det.reshape(-1, A_det.shape[-1])
            Z_2d = Z_det.reshape(-1, Z_det.shape[-1])
            B_2d = B_det.reshape(-1, B_det.shape[-2], B_det.shape[-1]).mean(dim=0)
            
            A_proj_2d, B_proj_2d, _, t_new = matmul_proj(
                A_2d, B_2d, Z_2d, t_init=t_init, alpha=ctx.alpha, g=ctx.g, num_steps=ctx.num_steps, omega=ctx.omega)
            
            A_proj = A_proj_2d.reshape(A_det.shape)
            B_proj = B_proj_2d.reshape(B_det.shape[-2], B_det.shape[-1]).expand(B_det.shape)
        else:
            A_proj, B_proj, _, t_new = matmul_proj(
                A_det, B_det, Z_det, t_init=t_init, alpha=ctx.alpha, g=ctx.g, num_steps=ctx.num_steps, omega=ctx.omega)

        # Update the cache for the next iteration
        if ctx.proj_cache is not None:
            ctx.proj_cache['t'] = t_new

        # Return Nones for proj_cache, alpha, g to match forward args
        return A_proj, B_proj, None, None, None, None, None


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

class MarginLossProjection(torch.autograd.Function):
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

        return new_logits, labels


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

class SumReluProjection(torch.autograd.Function):
    """
    PyTorch equivalent of PJAX sum_relu projection.
    Forward: relu(sum of inputs)
    Backward: projects inputs onto the sum-ReLU constraint graph.
    """
    @staticmethod
    def forward(ctx, *inputs):
        ctx.save_for_backward(*inputs)
        ctx.n_inputs = len(inputs)
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
