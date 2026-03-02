import time
import os
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "./torch_compile_cache"
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float32)
torch.manual_seed(42)

def bilinear_proj_fast_pt(a, b, z, num_steps=10):
    """
    Computes bilinear projection for vectors:
    p = a @ b
    q = |a|^2 + |b|^2
    """
    p = torch.dot(a, b)
    q = torch.dot(a, a) + torch.dot(b, b)

    def safe_newton_step(t):
        t2 = t * t
        one_minus_t2 = 1.0 - t2
        
        N = (1.0 + t2) * p + t * q
        f_val = (N / (one_minus_t2 * one_minus_t2)) - z
        
        N_prime = 2.0 * t * p + q
        f_prime_val = ((N_prime * one_minus_t2) + 4.0 * t * N) / (one_minus_t2 * one_minus_t2 * one_minus_t2)
        
        step = f_val / (f_prime_val + 1e-8)
        return torch.clamp(t - step, -0.999, 0.999)

    t = torch.tensor(0.0, device=a.device)
    for _ in range(num_steps):
        t = safe_newton_step(t)

    t2 = t * t
    denom = 1.0 - t2
    a_new = (a + t * b) / denom
    b_new = (b + t * a) / denom
    
    return a_new, b_new


# VMAP OVER BATCH AND DIMENSIONS
# PyTorch vmap:
# For A @ B = Z
# A is (Batch, M, K), B is (K, N), Z is (Batch, M, N)
# But let's look at pjax implementation: A is (M, K), B is (K, N), Z is (M, N)
# The atomic projection is `bilinear_matrix` which projects A_row and B_col onto Z_ij
# In pjax:
# a_buffer, projection_B = jax.vmap(proj_single, in_axes=(1, 0), out_axes=(0, 1))(B, z)
# proj_a = jnp.mean(a_buffer, axis=0)

@torch.compile
def bilinear_matrix_parr_pt(a, B, z, num_steps=10):
    """
    Project a vector `a` (shape K) and a matrix `B` (shape K, N) 
    onto a target vector `z` (shape N).
    
    This computes the bilinear projection of `a` and each column of `B` onto the 
    corresponding element of `z`.
    """
    # a: (K,)
    # B: (K, N)
    # z: (N,)
    
    # We want to use torch.vmap to run bilinear_proj_fast_pt across the columns of B and elements of Z
    # In PyTorch vmap syntax, we specify which dimension to map over (in_dims).
    # a has no batch dim (None), B is mapped over dim 1, z is mapped over dim 0
    # b_col is (K,), z_elem is scalar
    def proj_single(b_col, z_elem):
        return bilinear_proj_fast_pt(a, b_col, z_elem, num_steps)
    
    # vmap signature: vmap(func, in_dims=(None, 1, 0), out_dims=(0, 1))
    vmap_proj = torch.vmap(proj_single, in_dims=(1, 0), out_dims=(0, 1))
    
    # a_buffer: (N, K)
    # projection_B: (K, N)
    a_buffer, projection_B = vmap_proj(B, z)
    
    # Average the projected `a` vectors across the N targets
    proj_a = torch.mean(a_buffer, dim=0) # (K,)
    
    return proj_a, projection_B

@torch.compile
def matmul_proj_parr_pt(A, B, Z, num_steps=10):
    """
    Project A (M, K) and B (K, N) onto Z (M, N).
    """
    # A: (M, K)
    # B: (K, N)
    # Z: (M, N)
    
    # We map over the rows of A and Z. B is shared.
    def process_sample(A_row, Z_row):
        return bilinear_matrix_parr_pt(A_row, B, Z_row, num_steps)
    
    vmap_matmul = torch.vmap(process_sample, in_dims=(0, 0), out_dims=(0, 0))
    
    # A_buffer: (M, K)
    # B_buffer: (M, K, N)
    A_new, B_buffer = vmap_matmul(A, Z)
    
    # Average B across the M targets
    B_new = torch.mean(B_buffer, dim=0) # (K, N)
    
    return A_new, B_new

class MatMulProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        ctx.save_for_backward(A, B)
        return A @ B

    @staticmethod
    def backward(ctx, Z_target):
        A, B = ctx.saved_tensors
        A_proj, B_proj = matmul_proj_parr_pt(A, B, Z_target)
        return A_proj, B_proj

def test_matmul():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    M, K, N = 128, 256, 128
    A = torch.randn(M, K, device=device, requires_grad=True)
    B = torch.randn(K, N, device=device, requires_grad=True)
    
    # Forward Pass
    start_fwd = time.time()
    Output = MatMulProjection.apply(A, B)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t_fwd = time.time() - start_fwd
    
    # Target (e.g., some projection target)
    Z_target = Output + torch.randn_like(Output) * 0.1
    
    # Backward Pass (Projection) - compilation
    print("Compiling (first run)...")
    start_bwd = time.time()
    Output.backward(Z_target, retain_graph=True)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t_bwd_compile = time.time() - start_bwd
    
    A.grad = None
    B.grad = None
    
    print("Running timed projection...")
    start_bwd = time.time()
    Output.backward(Z_target)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    t_bwd = time.time() - start_bwd
    
    print(f"Forward time: {t_fwd * 1000:.2f} ms")
    print(f"Backward time (compile): {t_bwd_compile * 1000:.2f} ms")
    print(f"Backward time (compiled): {t_bwd * 1000:.2f} ms")
    
    # Check correctness mechanically
    print(f"A grad shape: {A.grad.shape}, expected {(M, K)}")
    print(f"B grad shape: {B.grad.shape}, expected {(K, N)}")

if __name__ == "__main__":
    test_matmul()
