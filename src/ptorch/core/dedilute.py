"""De-dilution pass-through card for Linear crossings (2026-07-10).

Fixes the generic input-demand attenuation measured at EVERY Linear card
(probe_outproj_deflation.py: transported displacement ~1e-4 of output
displacement at attn.proj ≈ mlp.proj ≈ mlp.fc): matmul_proj's consensus
averages the per-entry input proposals over N = out_features,

    A_proj = (alpha/N) * (A * sum_j inv_denom + t_inv_denom @ B^T),

so the transported term reaches the input diluted by 1/N while its
DIRECTION is fine (true-grad self-parallel share <= 0.12, healthy grad-twin
cosines at mlp/lm_head). This node sits on the INPUT tensor of a Linear and
undoes the mean in the backward target stream:

    d      = target - x                (consensus displacement, backward)
    d_perp = d - (d . x_hat) x_hat     (strip the radial self-rider: the
                                        (alpha/N)*sum inv_denom - 1 >= 0
                                        term scales with gain too and is
                                        exactly the self-inflation the
                                        deflate work fights; true demand is
                                        ~orthogonal to x rows anyway)
    d_new  = gain * d_perp             (gain = N: mean -> sum, the trunk
                                        analog of the QK/PV sum transport)
    cap    |d_new| per row at td_eps_lin * |x|   (CLAMP, not normalize —
                                        magnitude is honest after the gain;
                                        the window only guards outliers)
    return x + d_new

Forward is the identity; with DEDILUTE_LINEAR unset (default) the call
sites return x untouched — bit-identical legacy behavior. In gradient mode
(use_projections False) the backward passes the true gradient through
unchanged, so grad-twin references are unaffected.

Config keys (seeded from these env vars; see config.py):
  dedilute_linear     (DEDILUTE_LINEAR=1)      enable
  dedilute_gain_scale (DEDILUTE_GAIN_SCALE=f)  multiply the N gain by f (default 1.0)
"""
import torch

from ..config import config
from .ops import td_forward


class _DediluteFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gain):
        ctx.gain = gain
        ctx.use_proj = bool(config.use_projections)
        ctx.save_for_backward(x)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        if not ctx.use_proj:
            return g, None
        (x,) = ctx.saved_tensors
        td_forward(ctx)  # graph-scope TD floor passes through to x's producer
        d = g - x
        # Strip the radial self-rider before amplifying.
        xhat = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        d = d - (d * xhat).sum(-1, keepdim=True) * xhat
        d = d * ctx.gain
        if config.td_eps_lin > 0.0:
            cap = config.td_eps_lin * x.norm(dim=-1, keepdim=True)
            scale = (cap / d.norm(dim=-1, keepdim=True).clamp_min(1e-12))
            d = d * scale.clamp(max=1.0)
        return x + d, None


def dedilute(x, n_out: int):
    """Insert before a Linear with ``n_out`` output features. No-op unless
    DEDILUTE_LINEAR=1 (and always a forward no-op)."""
    if not config.dedilute_linear or not torch.is_grad_enabled():
        return x
    return _DediluteFn.apply(x, float(n_out) * config.dedilute_gain_scale)
