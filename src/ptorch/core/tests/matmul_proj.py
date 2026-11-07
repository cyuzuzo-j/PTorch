"""
Test file confirming the functioning of matmul_proj
"""
import torch
from .. import ops

def test_trivial_projection():
    A = torch.zeros(2, 3)
    B = torch.zeros(3, 4)
    Z = torch.zeros(2, 4)
    A_new, B_new, Z_new, t = ops.matmul_proj(A, B, Z)
    assert torch.allclose(A_new, A)
    assert torch.allclose(B_new, B)
    assert torch.allclose(Z_new, Z)
    assert torch.allclose(t, torch.zeros_like(t))

def test_projection_shape():
    M, K, N = 5, 4, 3
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    Z = torch.randn(M, N)
    
    A_new, B_new, Z_new, t = ops.matmul_proj(A, B, Z)
    
    assert A_new.shape == A.shape
    assert B_new.shape == B.shape
    assert Z_new.shape == Z.shape
    assert t.shape == (M, N)

def test_projection_moves_towards_target():
    torch.manual_seed(42)
    M, K, N = 3, 2, 3
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    Z = torch.randn(M, N)

    initial_diff = torch.norm(A @ B - Z)
    
    A_new, B_new, Z_new, t = ops.matmul_proj(A, B, Z, num_steps=10)
    
    final_diff = torch.norm(A_new @ B_new - Z_new)
    
    # After projection, the resulting matrices should be closer to satisfying A @ B = Z
    assert final_diff < initial_diff

def test_projection_idempotent_when_satisfied():
    # If the constraint is already satisfied, the projection should not change the matrices
    torch.manual_seed(42)
    M, K, N = 3, 4, 3
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    Z = A @ B  # Already satisfies constraint perfectly
    
    A_new, B_new, Z_new, t = ops.matmul_proj(A, B, Z, num_steps=5)
    
    # Since it already satisfies it, the update should be negligible
    assert torch.allclose(A_new, A, atol=1e-6)
    assert torch.allclose(B_new, B, atol=1e-6)

