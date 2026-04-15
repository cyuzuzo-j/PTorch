"""
Tests for relu_bilinear_proj_linf — ℓ∞ ReLU bilinear projection.
"""
import torch
import pytest
from .. import ops


def test_trivial_already_on_active_manifold():
    """When y = max(0, w^T x) > 0, eps* should be ~0."""
    torch.manual_seed(42)
    n = 5
    x = torch.randn(4, n)
    w = torch.randn(4, n)
    D = (w * x).sum(dim=-1)
    y = torch.relu(D)  # exactly on manifold

    x_p, w_p, y_p, eps = ops.relu_bilinear_proj_linf(x, w, y, gamma=1.0, num_steps=20)

    assert eps.shape == (4,)
    assert torch.allclose(eps, torch.zeros(4), atol=1e-4), f"Expected eps ≈ 0, got {eps}"
    assert torch.allclose(x_p, x, atol=1e-4)
    assert torch.allclose(w_p, w, atol=1e-4)


def test_trivial_already_on_inactive_manifold():
    """When y = 0 and w^T x <= 0, eps* should be ~0."""
    torch.manual_seed(42)
    n = 5
    x = torch.randn(4, n)
    w = -x  # ensure D = w^T x = sum(-x^2) < 0
    y = torch.zeros(4)

    x_p, w_p, y_p, eps = ops.relu_bilinear_proj_linf(x, w, y, gamma=1.0, num_steps=20)

    assert torch.allclose(eps, torch.zeros(4), atol=1e-4), f"Expected eps ≈ 0, got {eps}"


def test_shape_preserved():
    """Output shapes should match input shapes."""
    batch, n = 8, 10
    x = torch.randn(batch, n)
    w = torch.randn(batch, n)
    y = torch.randn(batch)

    x_p, w_p, y_p, eps = ops.relu_bilinear_proj_linf(x, w, y)

    assert x_p.shape == (batch, n)
    assert w_p.shape == (batch, n)
    assert y_p.shape == (batch,)
    assert eps.shape == (batch,)


def test_projection_reduces_gap():
    """After projection, constraint violation should be reduced."""
    torch.manual_seed(42)
    batch, n = 16, 8
    x = torch.randn(batch, n)
    w = torch.randn(batch, n)
    y = torch.randn(batch) * 2  # intentionally off manifold

    x_p, w_p, y_p, eps = ops.relu_bilinear_proj_linf(x, w, y, gamma=1.0, num_steps=20)

    # Check that projected y' is closer to relu(w'^T x')
    D_proj = (w_p * x_p).sum(dim=-1)
    relu_D_proj = torch.relu(D_proj)
    gap = (y_p - relu_D_proj).abs()

    # Projected points should approximately satisfy the constraint
    assert gap.max() < 0.5, f"Max constraint gap {gap.max():.4f} too large"


def test_active_regime():
    """When y is large and positive, the active branch (y' > 0) should be chosen."""
    torch.manual_seed(42)
    n = 4
    x = torch.ones(2, n)
    w = torch.ones(2, n)
    # D = n = 4, y = 5 > D, will use active UP branch
    y = torch.tensor([5.0, 5.0])

    x_p, w_p, y_p, eps = ops.relu_bilinear_proj_linf(x, w, y, gamma=1.0, num_steps=20)

    # Active regime: y' should be positive
    assert (y_p > 0).all(), f"Expected y' > 0, got {y_p}"


def test_inactive_regime():
    """When y is 0 and D is small, the inactive branch should be cheaper."""
    torch.manual_seed(42)
    n = 4
    x = torch.ones(2, n) * 0.1
    w = torch.ones(2, n) * 0.1
    # D = 0.04 (small positive), y = 0
    y = torch.tensor([0.0, 0.0])

    x_p, w_p, y_p, eps = ops.relu_bilinear_proj_linf(x, w, y, gamma=1.0, num_steps=20)

    # Inactive regime: y' should be 0, eps should be small
    assert (y_p == 0).all() or eps.max() < 0.2, f"Expected inactive regime, got y'={y_p}, eps={eps}"


def test_batch_vectorization():
    """Each sample in the batch should be projected independently."""
    torch.manual_seed(42)
    n = 6
    x = torch.randn(8, n)
    w = torch.randn(8, n)
    y = torch.randn(8)

    x_p, w_p, y_p, eps = ops.relu_bilinear_proj_linf(x, w, y, gamma=1.0, num_steps=20)

    # Run single-sample projection and compare
    for i in range(8):
        x_i = x[i:i+1]
        w_i = w[i:i+1]
        y_i = y[i:i+1]
        x_pi, w_pi, y_pi, eps_i = ops.relu_bilinear_proj_linf(x_i, w_i, y_i, gamma=1.0, num_steps=20)

        assert torch.allclose(eps[i], eps_i[0], atol=1e-4), \
            f"Sample {i}: batch eps={eps[i]:.6f} vs single eps={eps_i[0]:.6f}"


def test_gamma_scaling():
    """Higher gamma should penalize y-movement more, requiring larger eps."""
    torch.manual_seed(42)
    n = 4
    x = torch.randn(4, n)
    w = torch.randn(4, n)
    y = torch.randn(4) * 3  # off manifold

    _, _, _, eps_small = ops.relu_bilinear_proj_linf(x, w, y, gamma=0.5, num_steps=20)
    _, _, _, eps_large = ops.relu_bilinear_proj_linf(x, w, y, gamma=2.0, num_steps=20)

    # With gamma=2.0 (larger γ²), y is more constrained → eps should generally differ
    # This is a soft test: just verify they produce different results
    assert not torch.allclose(eps_small, eps_large, atol=1e-6), \
        "Different gamma values should produce different epsilon values"
