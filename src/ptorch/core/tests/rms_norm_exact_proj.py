"""
Test file confirming the functioning of rmsnorm_proj_exact (the joint L2
projection onto the RMSNorm graph manifold) and the config-dispatched
"exact" backward mode of RMSNormProjection.
"""
import math

import torch
import torch.nn.functional as F

from .. import ops
from ...config import config


def _manifold_objective(x_bar, x, z, g):
    """Distance of (x_bar, rmsnorm(x_bar)) to the data pair (x, z).

    For sigma = 0 rows (x_bar = 0) the manifold output is sqrt(n)*u for the
    optimal direction u, which this surrogate cannot see — callers should
    avoid degenerate rows or compare J values instead.
    """
    n = x.size(-1)
    norm = x_bar.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    z_bar = math.sqrt(n) * x_bar / norm
    return ((x_bar - x) ** 2).sum(-1) + g * ((z_bar - z) ** 2).sum(-1)


def test_exact_beats_or_ties_legacy():
    """The joint solve must never end with a worse objective than the legacy
    sequential reconstruction (it starts from the legacy candidate)."""
    torch.manual_seed(0)
    g = 1.0
    n = 8
    x = torch.randn(512, n)
    z = torch.randn(512, n) * torch.logspace(-3, 1, 512).unsqueeze(1)

    x_exact = ops.rmsnorm_proj_exact(x, z, g, 6)

    z_dir = math.sqrt(n) * z / (z.norm(dim=-1, keepdim=True) + 1e-8)
    sigma = (x * z_dir).mean(-1, keepdim=True)
    x_legacy = sigma * z_dir

    obj_exact = _manifold_objective(x_exact, x, z, g)
    obj_legacy = _manifold_objective(x_legacy, x, z, g)
    # Exclude sigma=0 rows where the surrogate objective is not faithful.
    valid = x_exact.norm(dim=-1) > 1e-9
    assert (obj_exact[valid] <= obj_legacy[valid] + 1e-5).all()


def test_exact_matches_theta_grid():
    """Compare against a dense brute-force scan of the reduced 1-D problem."""
    torch.manual_seed(1)
    g = 1.0
    n = 8
    sqrt_n = math.sqrt(n)
    x = torch.randn(256, n)
    z = torch.randn(256, n)

    x_exact = ops.rmsnorm_proj_exact(x, z, g, 6)

    a = x.norm(dim=-1, keepdim=True)
    e1 = x / a
    b1 = (z * e1).sum(-1, keepdim=True)
    r = z - b1 * e1
    b2 = r.norm(dim=-1, keepdim=True)
    e2 = r / b2.clamp_min(1e-12)
    thetas = torch.linspace(0, math.pi, 20001)
    ct, st = torch.cos(thetas), torch.sin(thetas)
    J = (a * ct).clamp_min(0) ** 2 + 2 * g * sqrt_n * (b1 * ct + b2 * st)
    idx = J.argmax(dim=1)
    j_brute = J.gather(1, idx.unsqueeze(1)).squeeze(1)

    # J value achieved by the solver's reconstruction
    sig = x_exact.norm(dim=-1, keepdim=True)
    ct_o = ((x_exact * e1).sum(-1, keepdim=True) / sig.clamp_min(1e-12)).clamp(-1, 1)
    st_o = (1 - ct_o.square()).clamp_min(0).sqrt()
    j_ours = ((a * ct_o).clamp_min(0) ** 2 + 2 * g * sqrt_n * (b1 * ct_o + b2 * st_o)).squeeze(1)

    valid = sig.squeeze(1) > 1e-9
    assert (j_ours[valid] >= j_brute[valid] - 1e-2).all()


def test_vanished_target_is_a_no_op():
    """A near-zero target must not be renormalized into a full-magnitude
    teaching signal (the legacy failure mode): x_bar should stay at x."""
    torch.manual_seed(2)
    x = torch.randn(16, 8)
    z = torch.randn(16, 8) * 1e-9
    x_bar = ops.rmsnorm_proj_exact(x, z, 1.0, 6)
    assert ((x_bar - x).norm() / x.norm()) < 1e-5


def test_backward_mode_dispatch():
    """RMSNormProjection.backward must route through the exact solver when
    config.rmsnorm_backward_mode == 'exact' and return a finite target."""
    torch.manual_seed(3)
    try:
        config.update("rmsnorm_backward_mode", "exact")
        x = torch.randn(4, 7, requires_grad=True)
        z = ops.RMSNormProjection.apply(x, 1e-5)
        z.backward(F.rms_norm(x.detach() + 0.1, (7,), eps=1e-5))
        assert torch.isfinite(x.grad).all()
    finally:
        config.update("rmsnorm_backward_mode", "legacy")


if __name__ == "__main__":
    test_exact_beats_or_ties_legacy()
    test_exact_matches_theta_grid()
    test_vanished_target_is_a_no_op()
    test_backward_mode_dispatch()
    print("All rmsnorm_proj_exact tests passed.")
