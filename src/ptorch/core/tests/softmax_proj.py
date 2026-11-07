"""
Test file confirming the functioning of SoftmaxProjection.

Covers:
  - forward equals softmax(x, dim=-1)
  - fixed-point: if z_target == softmax(x) then x_bar == x
  - one-hot target: argmax(x_bar) == argmax(z_target)
"""
import torch
import torch.nn.functional as F
from .. import ops


class _MockCtx:
    def __init__(self):
        self.saved_tensors = None

    def save_for_backward(self, *args):
        self.saved_tensors = args


def test_forward_matches_softmax():
    torch.manual_seed(0)
    x = torch.randn(4, 7)
    ctx = _MockCtx()
    z = ops.SoftmaxProjection.forward(ctx, x)
    assert torch.allclose(z, F.softmax(x, dim=-1), atol=1e-6)


def test_backward_fixed_point():
    """If z_target equals softmax(x), the projected input must equal x."""
    torch.manual_seed(1)
    x = torch.randn(3, 5)
    z_target = F.softmax(x, dim=-1)

    ctx = _MockCtx()
    ops.SoftmaxProjection.forward(ctx, x)
    x_bar, _ = ops.SoftmaxProjection.backward(ctx, z_target)

    assert torch.allclose(x_bar, x, atol=1e-5), (
        f"Fixed-point violated: max diff {(x_bar - x).abs().max().item():.2e}"
    )


def test_backward_one_hot_argmax():
    """One-hot target -> projected x must put its maximum on the same index."""
    torch.manual_seed(2)
    batch, C = 6, 8
    x = torch.randn(batch, C)
    labels = torch.randint(0, C, (batch,))
    z_target = F.one_hot(labels, num_classes=C).float()

    ctx = _MockCtx()
    ops.SoftmaxProjection.forward(ctx, x)
    x_bar, _ = ops.SoftmaxProjection.backward(ctx, z_target)

    assert torch.all(x_bar.argmax(dim=-1) == labels), (
        "Projected pre-activation should peak on the one-hot class"
    )


def test_backward_handles_negative_targets():
    """Negative target entries get clamped to eps; the projected logits stay finite."""
    torch.manual_seed(4)
    x = torch.randn(4, 6)
    z_target = torch.randn(4, 6)  # may contain negatives

    ctx = _MockCtx()
    ops.SoftmaxProjection.forward(ctx, x)
    x_bar, _ = ops.SoftmaxProjection.backward(ctx, z_target)

    assert torch.isfinite(x_bar).all()


def test_backward_shift_invariance():
    """Softmax is shift-invariant: shifting x by a scalar c must shift x_bar by c too."""
    torch.manual_seed(5)
    x = torch.randn(4, 7)
    z_target = F.softmax(torch.randn(4, 7), dim=-1)
    c = 1.7  # arbitrary per-row constant shift

    ctx_a = _MockCtx()
    ops.SoftmaxProjection.forward(ctx_a, x)
    x_bar_a, _ = ops.SoftmaxProjection.backward(ctx_a, z_target)

    ctx_b = _MockCtx()
    ops.SoftmaxProjection.forward(ctx_b, x + c)
    x_bar_b, _ = ops.SoftmaxProjection.backward(ctx_b, z_target)

    assert torch.allclose(x_bar_b, x_bar_a + c, atol=1e-5), (
        "Projection failed shift-invariance under uniform x shift"
    )


if __name__ == "__main__":
    test_forward_matches_softmax()
    test_backward_fixed_point()
    test_backward_one_hot_argmax()
    test_backward_handles_negative_targets()
    test_backward_shift_invariance()
    print("All SoftmaxProjection unit tests passed.")
