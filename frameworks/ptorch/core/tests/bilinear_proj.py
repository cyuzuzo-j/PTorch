"""
Test file confirming the functioning of bilinear projection
"""
import torch
from .. import ops

def test_trivial_projection():
    a = torch.tensor([0.0, 0.0])
    b = torch.tensor([0.0, 0.0])
    z = torch.tensor([0.0])
    a_new, b_new, z_new = ops._bilinear_proj_core(a, b, z)
    assert torch.allclose(a_new, a)
    assert torch.allclose(b_new, b)
    assert torch.allclose(z_new, z)

def test_projection_satisfies_constraint():
    # Use float64 because the large magnitudes of a, b (~10^4) and resulting
    # dot products (~10^8) exceed float32 precision capabilities (~7 decimal digits),
    # causing Newton's method steps to become noisy and preventing exact constraint satisfaction.
    a = torch.tensor([23413.8765, 1234.5678], dtype=torch.float64)
    b = torch.tensor([28358.24, 8465.2345], dtype=torch.float64)
    z = torch.tensor([1264684.0], dtype=torch.float64)
    a_new, b_new, z_new = ops._bilinear_proj_core(a, b, z)
    assert torch.allclose(a_new @ b_new.T, z_new)

def test_projection_idempotent():
    a = torch.tensor([23413.8765, 1234.5678])
    b = torch.tensor([28358.24, 8465.2345])
    z = torch.tensor([1264684.0])
    a_new, b_new, z_new = ops._bilinear_proj_core(a, b, z)
    a_final, b_final, z_final = ops._bilinear_proj_core(a_new, b_new, z_new)
    assert torch.allclose(a_final, a_new)
    assert torch.allclose(b_final, b_new)
    assert torch.allclose(z_final, z_new)

def test_projection_minimizes_distance():
    torch.manual_seed(42)
    a = torch.randn(10)
    b = torch.randn(10)
    z = torch.randn(1)
    
    # We use num_steps=50 to get a highly accurate stationary point
    a_new, b_new, z_new = ops._bilinear_proj_core(a, b, z, alpha=1.0, g=1.0, omega=1.0, num_steps=50)
    
    # Check that constraint is met (z_new = a_new * b_new)
    assert torch.allclose((a_new * b_new).sum(dim=-1), z_new, atol=1e-5)
    
    # Distance of the projection
    dist_proj = ((a_new - a)**2).sum() + ((b_new - b)**2).sum() + ((z_new - z)**2).sum()
    
    # Test random perturbations on the constraint manifold
    for _ in range(1000):
        # random perturbation
        delta_a = torch.randn(10) * 1e-3
        delta_b = torch.randn(10) * 1e-3
        
        a_pert = a_new + delta_a
        b_pert = b_new + delta_b
        
        # calculate z_pert such that the constraint is exactly satisfied
        z_pert = (a_pert * b_pert).sum(dim=-1)
        
        dist_pert = ((a_pert - a)**2).sum() + ((b_pert - b)**2).sum() + ((z_pert - z)**2).sum()
        
        # dist_pert should be strictly greater or equal to dist_proj
        # (with a tiny tolerance for floating point errors)
        assert dist_pert + 1e-6 >= dist_proj
