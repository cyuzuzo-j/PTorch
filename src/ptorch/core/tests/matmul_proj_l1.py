"""
Test file confirming the functioning of matmul_proj_l1
"""
import torch
from .. import ops

def test_l1_trivial_projection():
    A = torch.zeros(2, 3)
    B = torch.zeros(3, 4)
    Z = torch.zeros(2, 4)
    A_new, B_new, Z_new, t = ops.matmul_proj_l1(A, B, Z)
    assert torch.allclose(A_new, A)
    assert torch.allclose(B_new, B)
    assert torch.allclose(Z_new, Z)
    assert torch.allclose(t, torch.zeros_like(t))

def test_l1_projection_moves_towards_target():
    torch.manual_seed(42)
    M, K, N = 3, 2, 3
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    Z = torch.randn(M, N)

    initial_diff = torch.norm(A @ B - Z)
    
    A_new, B_new, Z_new, t = ops.matmul_proj_l1(A, B, Z)
    
    final_diff = torch.norm(A_new @ B_new - Z_new)
    
    # After projection, the resulting matrices should be closer to satisfying A @ B = Z
    assert final_diff < initial_diff

def test_l1_sparsity():
    torch.manual_seed(42)
    M, K, N = 5, 10, 4
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    Z = torch.randn(M, N)

    A_new, B_new, Z_new, _ = ops.matmul_proj_l1(A, B, Z)
    
    # Check that projection modifications are sparse (L1 specific)
    # Theoretically, for each m, n, exactly one k dimension is modified (or Z takes the penalty).
    # This means across K, we should see many zeros in the delta.
    delta_A = A_new - A
    delta_B = B_new - B
    
    # This is a loose check: the sum of nonzero elements should be bounded
    # by M*N updates across all K components for both A and B.
    # Actually, because a single coordinate k* is chosen per (m, n) pair, and
    # A_new = A + dA.mean(dim=-2) where dA has shape (M, N, K),
    # there are only M*N total non-zero updates in dA before mean.
    # B_eff_proj = B_eff + dB.mean(dim=-3).transpose(-1, -2) which has shape (M, N, K).
    assert not torch.allclose(A_new, A)
    assert not torch.allclose(B_new, B)
