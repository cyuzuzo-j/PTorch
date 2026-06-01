"""
Tests for SortProjection and the isotonic_projection helper.

Covers:
  - isotonic_projection: identity on sorted input, correct PAVA on simple cases
  - SortProjection.forward equals torch.sort(...).values along arbitrary dim
  - SortProjection.backward fixed point: z_target == sorted_x  =>  x_proj == x
  - Permutation invariance: any permutation of sorted_x as target yields x_proj == x
  - Off-manifold projection: x_proj reproduces the rank-matched midpoint after
    re-sorting, and equals scatter(indices, midpoint_sorted)
  - dim != -1 support
"""
import numpy as np
import torch

from frameworks.ptorch import config as ptorch_config
from frameworks.ptorch.core import ops


class _MockCtx:
    def __init__(self):
        self.saved_tensors = None
        self.dim = -1

    def save_for_backward(self, *args):
        self.saved_tensors = args


def _ensure_no_muon():
    ptorch_config.config.use_muon_activations = False


def test_isotonic_identity_on_sorted():
    x = torch.tensor([[1.0, 2.0, 2.0, 5.0, 7.0]])
    y = ops.isotonic_projection(x)
    assert torch.allclose(y, x)


def test_isotonic_pava_known_case():
    # [4, 4, 1, 1] -> all 2.5
    x = torch.tensor([[4.0, 4.0, 1.0, 1.0]])
    y = ops.isotonic_projection(x)
    assert torch.allclose(y, torch.full_like(x, 2.5))


def test_isotonic_pava_three_block():
    # [3, 1, 2] -> [2, 2, 2]
    x = torch.tensor([[3.0, 1.0, 2.0]])
    y = ops.isotonic_projection(x)
    assert torch.allclose(y, torch.full_like(x, 2.0))


def test_isotonic_batched():
    # Mixed rows: one already sorted, one needing PAVA.
    x = torch.tensor([[1.0, 2.0, 3.0],
                      [3.0, 1.0, 2.0]])
    y = ops.isotonic_projection(x)
    expected = torch.tensor([[1.0, 2.0, 3.0],
                             [2.0, 2.0, 2.0]])
    assert torch.allclose(y, expected)


def test_forward_matches_torch_sort_last_dim():
    torch.manual_seed(0)
    x = torch.randn(4, 7)
    ctx = _MockCtx()
    y = ops.SortProjection.forward(ctx, x, -1)
    assert torch.allclose(y, torch.sort(x, dim=-1).values)


def test_forward_matches_torch_sort_other_dim():
    torch.manual_seed(0)
    x = torch.randn(3, 5, 4)
    ctx = _MockCtx()
    y = ops.SortProjection.forward(ctx, x, 1)
    assert torch.allclose(y, torch.sort(x, dim=1).values)


def test_backward_fixed_point():
    """z_target == sorted_x  =>  x_proj == x exactly (we're already on the manifold)."""
    _ensure_no_muon()
    torch.manual_seed(1)
    x = torch.randn(4, 6)
    ctx = _MockCtx()
    sorted_x = ops.SortProjection.forward(ctx, x, -1)
    x_proj, _ = ops.SortProjection.backward(ctx, sorted_x)
    assert torch.allclose(x_proj, x, atol=1e-6), (x_proj - x).abs().max().item()


def test_backward_rank_order_of_target_matters():
    """
    z_target is rank-indexed (output is sorted ascending). Two targets that are
    permutations of each other must NOT collapse to the same x_proj — the
    rank carries real supervision. Concretely: z = sorted_x (already-feasible)
    is a fixed point; z = reversed(sorted_x) is NOT.
    """
    _ensure_no_muon()
    torch.manual_seed(2)
    x = torch.randn(3, 5)
    ctx = _MockCtx()
    sorted_x = ops.SortProjection.forward(ctx, x, -1)

    # Feasible target → no movement
    x_proj_fp, _ = ops.SortProjection.backward(ctx, sorted_x)
    assert torch.allclose(x_proj_fp, x, atol=1e-6)

    # Reversed-rank target → strong movement, and the projection compresses
    # the spread (PAVA on a Λ-shaped midpoint partially flattens it).
    rev = torch.flip(sorted_x, dims=[-1])
    x_proj_rev, _ = ops.SortProjection.backward(ctx, rev)
    assert not torch.allclose(x_proj_rev, x, atol=1e-3)
    # Spread must shrink: sort(x_proj) range < sort(x) range, per row.
    sx = torch.sort(x, dim=-1).values
    sp = torch.sort(x_proj_rev, dim=-1).values
    assert ((sp[..., -1] - sp[..., 0]) < (sx[..., -1] - sx[..., 0])).all()


def test_backward_off_manifold_matches_explicit_formula():
    """
    x_proj equals scatter(indices, PAVA((sorted_x + z_target) / 2)).
    """
    _ensure_no_muon()
    torch.manual_seed(3)
    x = torch.randn(2, 5)
    z_target = torch.randn(2, 5) * 2.0  # off manifold

    ctx = _MockCtx()
    sorted_x = ops.SortProjection.forward(ctx, x, -1)
    x_proj, _ = ops.SortProjection.backward(ctx, z_target)

    midpoint = (sorted_x + z_target) / 2.0
    v = ops.isotonic_projection(midpoint)

    _, indices = torch.sort(x, dim=-1)
    expected = torch.empty_like(x)
    expected.scatter_(-1, indices, v)
    assert torch.allclose(x_proj, expected, atol=1e-6)

    # sort(x_proj) must equal v (the ascending projection of the midpoint).
    assert torch.allclose(torch.sort(x_proj, dim=-1).values, v, atol=1e-6)


def test_backward_shape_preserved():
    _ensure_no_muon()
    torch.manual_seed(4)
    x = torch.randn(3, 4, 6)
    z_target = torch.randn(3, 4, 6)
    ctx = _MockCtx()
    ops.SortProjection.forward(ctx, x, -1)
    x_proj, _ = ops.SortProjection.backward(ctx, z_target)
    assert x_proj.shape == x.shape


def test_backward_dim_other_than_last():
    """Sorting along dim=1 in a 3-D tensor: same algebra, transposed."""
    _ensure_no_muon()
    torch.manual_seed(5)
    x = torch.randn(3, 5, 4)
    z_target = torch.randn(3, 5, 4)

    ctx = _MockCtx()
    sorted_x = ops.SortProjection.forward(ctx, x, 1)
    x_proj, _ = ops.SortProjection.backward(ctx, z_target)

    midpoint = (sorted_x + z_target) / 2.0
    # PAVA along dim=1: transpose to last, project, transpose back.
    v = ops.isotonic_projection(midpoint.transpose(1, -1)).transpose(1, -1)
    assert torch.allclose(torch.sort(x_proj, dim=1).values, v, atol=1e-6)


def test_apply_routes_through_autograd():
    """End-to-end: x.grad after .backward(z_target) equals the projected x."""
    _ensure_no_muon()
    torch.manual_seed(6)
    x = torch.randn(2, 5, requires_grad=True)
    z_target = torch.randn(2, 5)

    y = ops.SortProjection.apply(x, -1)
    y.backward(z_target)

    # Reproduce expected projection
    with torch.no_grad():
        sorted_x = torch.sort(x, dim=-1).values
        midpoint = (sorted_x + z_target) / 2.0
        v = ops.isotonic_projection(midpoint)
        _, indices = torch.sort(x, dim=-1)
        expected = torch.empty_like(x)
        expected.scatter_(-1, indices, v)

    assert x.grad.shape == x.shape
    assert torch.allclose(x.grad, expected, atol=1e-6)
