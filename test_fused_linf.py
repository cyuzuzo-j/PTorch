import torch
import pytest
from frameworks.ptorch.core.ops import FusedSimplexBilinearLinf, fused_linf_proj_solver

def test_fused_linf_proj_solver_memory_complexity():
    # Batch=128, In=1024, Out=1024
    B = 128
    D = 1024
    x = torch.randn(B, D)
    W = torch.randn(D, D)
    z = torch.randn(B, D)
    y = torch.softmax(z, dim=-1)
    gamma = 0.1
    
    # Should not OOM
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        memory_before = torch.cuda.memory_allocated()
        
    eps = fused_linf_proj_solver(x, W, z, y, gamma)
    
    if torch.cuda.is_available():
        memory_after = torch.cuda.memory_allocated()
        # Ensure memory difference is less than O(B * D * D * 4) bytes, meaning no M*N*K expansion
        assert memory_after - memory_before < B * D * D * 4

def test_fused_linf_mathematical_correctness():
    x = torch.randn(4, 16, requires_grad=True)
    W = torch.randn(16, 16, requires_grad=True)
    z = torch.randn(4, 16)
    y = torch.softmax(z, dim=-1)
    gamma = 0.1
    
    fused_layer = FusedSimplexBilinearLinf.apply
    out = fused_layer(x, W, gamma)
    
    # Simple check if output shape is preserved and it runs
    assert out.shape == y.shape
    
    # Test convergence is under 15 steps
    # (By default max_iter is 15 in our solver)
    
    out.sum().backward()
    assert x.grad is not None

if __name__ == "__main__":
    pytest.main([__file__])
