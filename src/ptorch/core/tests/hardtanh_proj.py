"""
Test file confirming the functioning of HardtanhProjection (the DyT-style
normalization-free layer's exact 3-branch graph projection).
"""
import torch

from .. import ops


def test_forward():
    x = torch.tensor([-10.0, -2.0, 0.0, 1.0, 10.0])
    y = ops.HardtanhProjection.apply(x, 0.5)
    assert torch.allclose(y, torch.tensor([-1.0, -1.0, 0.0, 0.5, 1.0]))


def test_backward_matches_brute_force():
    """The returned input target must achieve the global minimum distance to
    the graph of y = clamp(alpha*x, -1, 1) (compare objective values — at
    branch-boundary ties the argmin is not unique)."""
    torch.manual_seed(0)
    alpha = 0.5
    xs = torch.randn(2000) * 4
    zs = torch.randn(2000) * 2

    x_in = xs.clone().requires_grad_(True)
    y = ops.HardtanhProjection.apply(x_in, alpha)
    y.backward(zs)
    x_proj = x_in.grad

    def objective(xc):
        yc = torch.clamp(alpha * xc, -1.0, 1.0)
        return (xc - xs) ** 2 + (yc - zs) ** 2

    grid = torch.linspace(-12, 12, 200001)
    y_grid = torch.clamp(alpha * grid, -1.0, 1.0)
    dist = (grid[None, :] - xs[:, None]) ** 2 + (y_grid[None, :] - zs[:, None]) ** 2
    brute_obj = dist.min(dim=1).values

    gap = objective(x_proj) - brute_obj
    assert float(gap.max()) < 1e-3


def test_fixed_point():
    """If the target equals the forward output, the input must not move."""
    torch.manual_seed(1)
    x = (torch.randn(64) * 3).requires_grad_(True)
    y = ops.HardtanhProjection.apply(x, 0.5)
    y.backward(y.detach())
    inside = x.detach().abs() < 2.0  # interior of the linear branch
    assert torch.allclose(x.grad[inside], x.detach()[inside], atol=1e-6)


if __name__ == "__main__":
    test_forward()
    test_backward_matches_brute_force()
    test_fixed_point()
    print("All HardtanhProjection tests passed.")
