import torch
import torch.nn.functional as F
from .. import config


def bilinear_proj_fast_pt(a, b, z, num_steps=10):
    """
    Batched projection onto bilinear function graph using Newton's method.
    Supports arbitrary leading batch dimensions.

    a: (*, K)  -- vectors
    b: (*, K)  -- vectors
    z: (*,)    -- scalar targets

    Returns a_new (*, K), b_new (*, K)
    """
    # p = <a, b>, q = ||a||^2 + ||b||^2   — sum over last axis keeps batch dims
    p = (a * b).sum(dim=-1)                         # (*,)
    q = (a * a).sum(dim=-1) + (b * b).sum(dim=-1)   # (*,)

    t = torch.zeros_like(p)                          # (*,)

    for _ in range(num_steps):
        t2 = t * t
        one_minus_t2 = 1.0 - t2

        N = (1.0 + t2) * p + t * q
        f_val = (N / (one_minus_t2 * one_minus_t2)) - z

        N_prime = 2.0 * t * p + q
        f_prime_val = ((N_prime * one_minus_t2) + 4.0 * t * N) / (one_minus_t2 ** 3)

        step = f_val / (f_prime_val + 1e-8)
        t = torch.clamp(t - step, -0.999, 0.999)

    t2 = t * t
    denom = 1.0 - t2

    # Unsqueeze for broadcasting with (*, K) tensors
    t_k = t.unsqueeze(-1)          # (*, 1)
    denom_k = denom.unsqueeze(-1)  # (*, 1)

    a_new = (a + t_k * b) / denom_k
    b_new = (b + t_k * a) / denom_k

    return a_new, b_new


def bilinearMatrix_seq(a, B, z, num_steps=10):
    """
    Inner scan: project vector a against matrix B with target z.
    a: (K,), B: (K, N), z: (N,)
    Sequentially iterates over columns of B (cyclic projection — a_curr carries).
    Returns a_final (K,), B_proj (K, N)

    Pre-allocates output tensor instead of Python list + torch.stack.
    """
    K, N = B.shape
    B_proj = torch.empty_like(B)   # (K, N) pre-allocated
    a_curr = a                     # (K,)

    for i in range(N):
        # Use unsqueeze(0) → (1, K) to match batched API, squeeze back
        a_new, b_new = bilinear_proj_fast_pt(
            a_curr.unsqueeze(0), B[:, i].unsqueeze(0), z[i].unsqueeze(0), num_steps
        )
        a_curr = a_new.squeeze(0)
        B_proj[:, i] = b_new.squeeze(0)

    return a_curr, B_proj


def bilinearMatrix_parr(a, B, z, num_steps=10):
    """
    Fully vectorized parallel projection — NO Python loop.
    a: (K,), B: (K, N), z: (N,)
    Projects each column of B independently against the SAME a.
    Returns a_proj (K,), B_proj (K, N)
    """
    N = B.shape[1]

    # Broadcast a → (N, K),  transpose B → (N, K),  z is already (N,)
    a_batch = a.unsqueeze(0).expand(N, -1)   # (N, K)
    b_batch = B.T                             # (N, K)

    a_projs, b_projs = bilinear_proj_fast_pt(a_batch, b_batch, z, num_steps)
    # a_projs: (N, K), b_projs: (N, K)

    a_proj = a_projs.mean(dim=0)   # (K,)
    B_proj = b_projs.T             # (K, N)

    return a_proj, B_proj

@torch.compile(mode="reduce-overhead")
def matmul_proj_seq_pt(A, B, Z, num_steps=10):
    """
    Project onto matrix multiplication constraint A @ B = Z.
    A: (M, K), B: (K, N), Z: (M, N).

    Outer scan over rows of A (sequential — B_curr is carried).
    Inner projection uses bilinearMatrix_parr (vectorized over columns).

    Pre-allocates output tensor instead of Python list + torch.stack.
    """
    M = A.shape[0]
    B_curr = B.clone()
    A_proj = torch.empty_like(A)   # (M, K) pre-allocated

    for i in range(M):
        a_proj, B_curr = bilinearMatrix_parr(A[i], B_curr, Z[i], num_steps)
        A_proj[i] = a_proj

    return A_proj, B_curr


# ─── autograd.Function wrappers ───────────────────────────────────────────────

class MatMulProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        ctx.save_for_backward(A, B)
        return A @ B

    @staticmethod
    def backward(ctx, Z_target):
        A, B = ctx.saved_tensors
        A_proj, B_proj = matmul_proj_seq_pt(A, B, Z_target)
        return A_proj, B_proj

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
