"""Block-boundary demand governor (2026-07-11).

The corrected direction/amplitude map (attenuation-fix campaign,
grad_twin_checkpoint.py with the _CleanAdd reference shim) showed the
projection demand is healthy in the upper network (|dY|/|Y| ~ 1-4e-2,
cos(dY, -g) 0.74-0.99) but blows out at block0: relative amplitude
0.5-2.2 (50-220% of the signal) co-located with the direction collapse
to cos ~0.30. A demand that large is outside the local solve's trust
region — the per-card consensus linearizes around the current
activations, so an out-of-regime demand converts magnitude error into
DIRECTION error before the optimizer ever sees a pseudo-grad (Muon's
polar factor then faithfully applies the corrupted subspace at full
unit step).

This node sits on the residual stream at a block boundary. Forward is
the identity. In projection backward it clamps the arriving demand into
the trust region where the solve below is valid:

    d     = target - x                     (arriving displacement)
    cap   = tau * |x|   per row            (GOVERNOR_TAU, default 0.05:
                                            just above the healthy band,
                                            ~10-40x cut at block0)
    d     = d * min(1, cap / |d|)          (DAMP only - never amplify,
                                            direction untouched)
    return x + d

Opposite sign of every falsified amplitude fix: those added magnitude
and lost to drift; this removes out-of-regime magnitude to protect
direction quality — the only axis that ever won.

With DEMAND_GOVERNOR unset (default) call sites return x untouched —
bit-identical legacy behavior. In gradient mode (use_projections False)
the backward passes the true gradient through unchanged.

Config keys (seeded from these env vars; see config.py):
  demand_governor     (DEMAND_GOVERNOR=1)      enable
  governor_tau        (GOVERNOR_TAU=f)         trust-region width (default 0.05)
  governor_boundaries (GOVERNOR_BOUNDARIES=0,1) comma list of governed boundaries; the card
                             before block i governs demand flowing DOWN
                             past that boundary (default "0,1": boundary
                             1 protects block0's internals, boundary 0
                             protects the embedding)
"""
import torch

from ..config import config
from .ops import td_forward


class _GovernorFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.use_proj = bool(config.use_projections)
        ctx.tau = config.governor_tau
        ctx.save_for_backward(x)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        if not ctx.use_proj:
            return g
        (x,) = ctx.saved_tensors
        td_forward(ctx)  # graph-scope TD floor passes through to x's producer
        d = g - x
        cap = ctx.tau * x.norm(dim=-1, keepdim=True)
        scale = (cap / d.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1.0)
        return x + d * scale


def govern(x, boundary: int):
    """Insert on the residual stream just before block ``boundary``.
    No-op unless DEMAND_GOVERNOR=1 and ``boundary`` is governed (and
    always a forward no-op)."""
    if (not config.demand_governor or boundary not in config.governor_boundaries
            or not torch.is_grad_enabled()):
        return x
    return _GovernorFn.apply(x)
