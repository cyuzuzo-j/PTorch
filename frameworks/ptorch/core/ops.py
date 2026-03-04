import torch
import torch.nn.functional as F
import warnings
from .. import config

# Suppress pin_memory warning when no accelerator is available
warnings.filterwarnings("ignore", message=".*pin_memory.*")

def _bilinear_proj_core(a, b, z, alpha=1.0, g=1.0, omega=1.0, num_steps=10):
    """
    Batched projection onto bilinear function graph using Newton's method.
    Pure tensor computation — no dynamic control flow.

    a: (*, K)      -- vectors (inputs)
    b: (*, K)      -- vectors (weights)
    z: (*,)        -- scalar targets (Omega * y)
    alpha: float   -- stiffness for weights  (penalty on ||w' - w||^2)
    g: float       -- stiffness for targets  (penalty on (y' - y)^2)
    omega: float   -- constraint multiplier for the target
    num_steps: int -- number of Newton iterations

    Returns a_new (*, K), b_new (*, K), z_new (*,)
    """
    p = (a * b).sum(dim=-1)            # (*,)
    qa = (a * a).sum(dim=-1)           # (*,)
    qb = (b * b).sum(dim=-1)           # (*,)
    
    # Effective q incorporating alpha
    q_eff = qa + alpha * qb            # (*,)
    
    t = torch.zeros_like(p)            # (*,)
    max_t = 0.999 * (alpha ** 0.5)
    
    # The penalty scaling term for the target z
    # Derived from Omega^2 / g^2  (g is independent of alpha)
    target_penalty = (omega ** 2) / (g ** 2)

    for _ in range(num_steps):
        t2 = t * t
        alpha_minus_t2 = alpha - t2

        N = alpha * (p * (alpha + t2) + t * q_eff)
        
        # Added the linear target_penalty term to f_val
        f_val = (N / (alpha_minus_t2 * alpha_minus_t2)) - z + t * target_penalty

        N_prime = alpha * (2.0 * t * p + q_eff)
        
        # Added the derivative of the linear term to f_prime_val
        f_prime_val = ((N_prime * alpha_minus_t2) + 4.0 * t * N) / (alpha_minus_t2 ** 3) + target_penalty

        step = f_val / (f_prime_val + 1e-8)
        t = torch.clamp(t - step, -max_t, max_t)

    t2 = t * t
    denom = alpha - t2

    t_k = t.unsqueeze(-1)          # (*, 1)
    denom_k = denom.unsqueeze(-1)  # (*, 1)

    # Reconstruct updated tensors
    a_new = alpha * (a + t_k * b) / denom_k
    b_new = (alpha * b + t_k * a) / denom_k
    
    # Projected target: z' = z - t * (omega^2 / g^2)
    z_new = z - t * target_penalty
    
    return a_new, b_new, z_new

def bilinearMatrix_seq(a, B, z):
    """
    Inner scan: project vector a against matrix B with target z.
    a: (K,), B: (K, N), z: (N,)
    Sequentially iterates over columns of B (cyclic projection — a_curr carries).
    Returns a_final (K,), B_proj (K, N)
    """
    K, N = B.shape
    b_cols = []
    a_curr = a

    for i in range(N):
        a_new, b_new, _z_new = _bilinear_proj_core(
            a_curr.unsqueeze(0), B[:, i].unsqueeze(0), z[i].unsqueeze(0)
        )
        a_curr = a_new.squeeze(0)
        b_cols.append(b_new.squeeze(0))

    B_proj = torch.stack(b_cols, dim=1)  # (K, N)
    return a_curr, B_proj


def bilinearMatrix_parr(a, B, z, alpha=1.0, g=1.0):
    """
    Fully vectorized parallel projection — NO Python loop.
    a: (K,), B: (K, N), z: (N,)
    alpha: float -- weight stiffness passed to _bilinear_proj_core
    g: float     -- target stiffness passed to _bilinear_proj_core
    Projects each column of B independently against the SAME a.
    Returns a_proj (K,), B_proj (K, N), z_proj (N,)
    """
    N = B.shape[1]

    a_batch = a.unsqueeze(0).expand(N, -1)          # (N, K)
    b_batch = B.t().contiguous()                     # (N, K)

    a_projs, b_projs, z_projs = _bilinear_proj_core(a_batch, b_batch, z, alpha=alpha, g=g)

    a_proj = a_projs.mean(dim=0)                     # (K,)
    B_proj = b_projs.t().contiguous()                # (K, N)

    return a_proj, B_proj, z_projs

@torch.compile(dynamic=True)
def matmul_proj_seq_pt(A, B, Z, alpha=1.0, g=1.0):
    """
    Project onto matrix multiplication constraint A @ B = Z.
    A: (M, K), B: (K, N), Z: (M, N).
    alpha: float -- weight stiffness forwarded to _bilinear_proj_core
    g: float     -- target stiffness forwarded to _bilinear_proj_core

    Outer scan over rows of A (sequential — B_curr is carried).
    Inner projection uses bilinearMatrix_parr (vectorized over columns).
    Returns A_proj (M, K), B_proj (K, N), Z_proj (M, N)
    """
    M = A.shape[0]
    B_curr = B.clone()
    a_rows = []
    z_rows = []

    for i in range(M):
        a_proj, B_curr, z_proj = bilinearMatrix_parr(A[i], B_curr, Z[i], alpha=alpha, g=g)
        a_rows.append(a_proj)
        z_rows.append(z_proj)

    A_proj = torch.stack(a_rows, dim=0)  # (M, K)
    Z_proj = torch.stack(z_rows, dim=0)  # (M, N)
    return A_proj, B_curr, Z_proj

def matmul_proj_parr_pt(A, B, Z, alpha=1.0, g=1.0):
    """
    Fully parallel projection onto matrix multiplication constraint A @ B = Z.
    A: (M, K), B: (K, N), Z: (M, N).
    alpha: float -- weight stiffness
    g: float     -- target stiffness

    Unlike matmul_proj_seq_pt, B is NOT carried between rows — each row
    projects independently against the SAME original B.  The M projected
    copies of B are averaged at the end.
    Returns A_proj (M, K), B_proj (K, N), Z_proj (M, N)
    """
    M = A.shape[0]
    a_rows = []
    B_projs = []
    z_rows = []

    for i in range(M):
        a_proj, B_proj_i, z_proj = bilinearMatrix_parr(A[i], B.clone(), Z[i], alpha=alpha, g=g)
        a_rows.append(a_proj)
        B_projs.append(B_proj_i)
        z_rows.append(z_proj)

    A_proj = torch.stack(a_rows, dim=0)                # (M, K)
    B_proj = torch.stack(B_projs, dim=0).mean(dim=0)   # (K, N) — average over rows
    Z_proj = torch.stack(z_rows, dim=0)                # (M, N)
    return A_proj, B_proj, Z_proj
    
@torch.compile(dynamic=True)
def matmul_proj_exact_pt(A, B, Z, alpha=1.0, g=1.0):
    """
    Exact independent bilinear projection for A @ B = Z.
    A: (M, K), B: (K, N), Z: (M, N).

    Every dot product z_ij = a_i · b_j gets its own independent bilinear
    projection. Shared variables are then reconciled by consensus averaging:
      - A_proj[i] = mean over j of the M×N individual a_i proposals
      - B_proj[:,j] = mean over i of the M×N individual b_j proposals

    This is equivalent to pjax.matmul_slower / matmul_proj_exact but in a
    single fully-vectorised pass — no Python loops.

    Returns A_proj (M, K), B_proj (K, N), Z_proj (M, N).
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Shape mismatch: A is ({M},{K}), B is ({K},{N})"

    # --- Expand all (i,j) pairs in one shot ---
    # a_i replicated for each column j:  (M, N, K)
    A_exp = A.unsqueeze(1).expand(M, N, K)   # (M, N, K)
    # b_j replicated for each row i:     (M, N, K)
    B_exp = B.t().unsqueeze(0).expand(M, N, K)   # (M, N, K)  — B.t() is (N,K)
    Z_flat = Z  # (M, N)

    # Flatten to (M*N, K) and (M*N,) for a single batched call
    A_flat = A_exp.reshape(M * N, K)
    B_flat = B_exp.reshape(M * N, K)
    Z_flat_1d = Z_flat.reshape(M * N)

    a_projs, b_projs, z_projs = _bilinear_proj_core(
        A_flat, B_flat, Z_flat_1d, alpha=alpha, g=g
    )
    # a_projs: (M*N, K),  b_projs: (M*N, K),  z_projs: (M*N,)

    # Reshape back to (M, N, K)
    A_proposals = a_projs.reshape(M, N, K)   # proposal of a_i from each j
    B_proposals = b_projs.reshape(M, N, K)   # proposal of b_j from each i

    # Consensus: average over the "other" index
    A_proj = A_proposals.mean(dim=1)          # (M, K) — average over N proposals
    B_proj = B_proposals.mean(dim=0).t()      # (K, N) — average over M proposals, then transpose
    Z_proj = z_projs.reshape(M, N)

    return A_proj, B_proj, Z_proj


@torch.compile(dynamic=True)
def matmul_proj_iterative_pt(A, B, Z, alpha=1.0, g=1.0, num_iters=5):
    """
    Iterated projection onto matrix multiplication constraint A @ B = Z.
    A: (M, K), B: (K, N), Z: (M, N).
    alpha: float    -- weight stiffness
    g: float        -- target stiffness
    num_iters: int  -- number of full sequential passes

    Each iteration runs a full sequential scan (matmul_proj_seq_pt) using
    the ORIGINAL (A, B, Z) as the reference but the CURRENT projected
    (A_curr, B_curr) as starting point.  The target Z remains fixed.

    This is cyclic projection / Gauss-Seidel with multiple sweeps,
    converging toward the true joint projection.
    Returns A_proj (M, K), B_proj (K, N), Z_proj (M, N)
    """
    A_curr = A.clone()
    B_curr = B.clone()
    Z_proj = Z.clone()

    for iteration in range(num_iters):
        M = A_curr.shape[0]
        a_rows = []
        z_rows = []

        for i in range(M):
            a_proj, B_curr, z_proj = bilinearMatrix_parr(
                A_curr[i], B_curr, Z[i], alpha=alpha, g=g
            )
            a_rows.append(a_proj)
            z_rows.append(z_proj)

        A_curr = torch.stack(a_rows, dim=0)
        Z_proj = torch.stack(z_rows, dim=0)

    return A_curr, B_curr, Z_proj

# ─── autograd.Function wrappers ───────────────────────────────────────────────

class MatMulProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, alpha=1.0, g=1.0, num_iters=1):
        ctx.save_for_backward(A, B)
        ctx.alpha = alpha
        ctx.g = g
        ctx.num_iters = num_iters
        return A @ B

    @staticmethod
    def backward(ctx, Z_target):
        A, B = ctx.saved_tensors
        A_det = A.detach()
        B_det = B.detach()
        Z_det = Z_target.detach()
        if ctx.num_iters <= 1:
            A_proj, B_proj, _Z_proj = matmul_proj_seq_pt(
                A_det, B_det, Z_det, alpha=ctx.alpha, g=ctx.g)
        else:
            A_proj, B_proj, _Z_proj = matmul_proj_iterative_pt(
                A_det, B_det, Z_det, alpha=ctx.alpha, g=ctx.g, num_iters=ctx.num_iters)
        return A_proj, B_proj, None, None, None

class MatMulExactProjection(torch.autograd.Function):
    """
    Exact per-dot-product bilinear projection for A @ B = Z.
    Uses matmul_proj_exact_pt in backward: fully vectorised, no Python loops.
    Equivalent to matmul_slower in pjax — better convergence at cost of memory.
    """
    @staticmethod
    def forward(ctx, A, B, alpha=1.0, g=1.0):
        ctx.save_for_backward(A, B)
        ctx.alpha = alpha
        ctx.g = g
        return A @ B

    @staticmethod
    def backward(ctx, Z_target):
        A, B = ctx.saved_tensors
        A_det = A.detach()
        B_det = B.detach()
        Z_det = Z_target.detach()

        # Handle batched inputs: collapse leading dims to 2D
        if A_det.ndim > 2:
            batch_shape = A_det.shape[:-2]
            A_2d = A_det.reshape(-1, A_det.shape[-1])
            Z_2d = Z_det.reshape(-1, Z_det.shape[-1])
            A_proj_2d, B_proj, _ = matmul_proj_exact_pt(
                A_2d, B_det, Z_2d, alpha=ctx.alpha, g=ctx.g)
            A_proj = A_proj_2d.reshape(A_det.shape)
        else:
            A_proj, B_proj, _ = matmul_proj_exact_pt(
                A_det, B_det, Z_det, alpha=ctx.alpha, g=ctx.g)

        return A_proj, B_proj, None, None

class MSEProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, predictions, targets):
        ctx.save_for_backward(predictions, targets)
        return F.mse_loss(predictions, targets)

    @staticmethod
    def backward(ctx, z):
        predictions, targets = ctx.saved_tensors
        out = (predictions + targets) / 2
        return out, out

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

class CrossEntropyProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels):
        ctx.save_for_backward(logits, labels)
        return logits

    @staticmethod
    def backward(ctx, z_target):
        logits, labels = ctx.saved_tensors
        lmbda = 1.0
        steps = 5

        x = logits
        for _ in range(steps):
            x = x + lmbda * (labels - F.softmax(x, dim=-1))

        return x, labels


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
