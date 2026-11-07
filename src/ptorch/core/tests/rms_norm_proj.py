"""
Test file confirming the functioning of RMSNormProjection.
"""
import torch
import torch.nn.functional as F
import math
from .. import ops

class _MockCtx:
    def __init__(self):
        self.saved_tensors = None

    def save_for_backward(self, *args):
        self.saved_tensors = args

def test_forward_matches_rmsnorm():
    torch.manual_seed(0)
    x = torch.randn(4, 7)
    ctx = _MockCtx()
    z = ops.RMSNormProjection.apply(x, 1e-5)
    assert torch.allclose(z, F.rms_norm(x, (x.size(-1),), eps=1e-5), atol=1e-6)

def test_backward_fixed_point():
    """If z_target equals RMSNorm(x), the projected input must equal x."""
    torch.manual_seed(1)
    x = torch.randn(3, 5, requires_grad=True)
    z_target = F.rms_norm(x, (x.size(-1),), eps=1e-5)

    z = ops.RMSNormProjection.apply(x, 1e-5)
    z.backward(z_target)
    x_bar = x.grad

    assert torch.allclose(x_bar, x, atol=1e-5), (
        f"Fixed-point violated: max diff {(x_bar - x).abs().max().item():.2e}"
    )

def test_backward_hypersphere():
    """The implicit z_bar must lie on the hypersphere sum(z_i^2) = n."""
    torch.manual_seed(2)
    n = 6
    x = torch.randn(4, n, requires_grad=True)
    z_target = torch.randn(4, n) * 10  # Arbitrary target

    z = ops.RMSNormProjection.apply(x, 1e-5)
    z.backward(z_target)
    x_bar = x.grad

    z_target_norm = torch.linalg.norm(z_target, dim=-1, keepdim=True)
    z_bar = math.sqrt(n) * (z_target / (z_target_norm + 1e-8))
    
    z_bar_sq_sum = (z_bar ** 2).sum(dim=-1)
    assert torch.allclose(z_bar_sq_sum, torch.tensor(float(n), dtype=x.dtype), atol=1e-4)

    sigma_bar = (x * z_bar).mean(dim=-1, keepdim=True)
    assert torch.allclose(x_bar, sigma_bar * z_bar, atol=1e-5)

if __name__ == "__main__":
    test_forward_matches_rmsnorm()
    test_backward_fixed_point()
    test_backward_hypersphere()
    print("All RMSNormProjection unit tests passed.")