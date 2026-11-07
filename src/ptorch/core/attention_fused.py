"""Shared pieces of the fused causal-attention projection node (PTORCH_FUSED_ATTN=1,
default off; the node is FusedCausalAttentionProjectionV2 in core/attention_native.py).

The node replaces the legacy attention core of MultiheadAttention

    GQA-expand -> QK MatMulProjection -> MaskedAddProjection
                -> SoftmaxProjection / SparsemaxProjection -> PV MatMulProjection

with a single autograd Function that saves only q, k, v (KV-head shape), one
row statistic (lse / τ) and y, recomputes the (B, H, T, T) pattern in backward,
and runs the legacy backward chain inline. Forward is bit-identical to the
legacy composition (same op order); backward targets agree to fp32 rounding.

TD repeaters: the legacy softmax-node repeater lifts the residual of the
finite logits to min(floor, td_eps_lin·‖logits‖); the mask-node repeater that
follows lifts the same residual to a window at least as large, so the fused
node applies only the latter (ops.td_repeat_target on the pre-mask logits).

This module holds the config envelope, the causal-mask cache and the compiled
softmax backward kernels (also used by ops.SoftmaxProjection).
"""

import os

import torch

from ..config import config
from . import ops

# PTORCH_COMPILE_DYNAMIC: 1 (default, symbolic shapes) / 0 (static-shape
# specialization; faster kernels when shapes are constant, recompiles per
# distinct shape) / auto (torch default).
_TC_DYNAMIC = {"1": True, "0": False, "auto": None}[os.environ.get("PTORCH_COMPILE_DYNAMIC", "1")]


def fused_attention_enabled() -> bool:
    """Config envelope of the fused node; read per forward call so scripts can
    flip config at runtime. Graph-scope / td_row / vector-mode TD, muon
    activations and DIAG need the per-node chain; the log-space softmax target
    modes other than rel_floor have no fused twin."""
    return (config.fused_attn and config.use_projections and ops.DIAG is None
            and not config.use_muon_activations
            and (config.attention_activation == "sparsemax"
                 or config.softmax_target_mode in ("jacobian", "rel_floor"))
            and (config.td_lambda == 0.0 or (config.td_mode == "norm"
                                             and config.td_scope == "global"
                                             and not config.td_row)))


# ── causal mask cache ─────────────────────────────────────────────────────────
# Float -inf mask (added to the logits in forward, matching the legacy
# MaskedAddProjection input bit-for-bit) and its boolean visibility twin, both
# (group·T, T): `group` stacks the mask for GQA heads folded into rows. Built
# outside inference mode: it may be created during eval and reused by a
# training forward.
_MASK_CACHE: dict = {}


def _causal_masks(seqlen: int, device: torch.device, dtype: torch.dtype, group: int = 1):
    key = (seqlen, device, dtype, group)
    masks = _MASK_CACHE.get(key)
    if masks is None:
        with torch.inference_mode(False):
            visible = torch.ones(seqlen, seqlen, device=device, dtype=torch.bool).tril_().repeat(group, 1)
            neg_inf = torch.zeros(visible.shape, device=device, dtype=dtype).masked_fill_(~visible, float("-inf"))
        _MASK_CACHE[key] = masks = (neg_inf, visible)
    return masks


# ── compiled backward impls ───────────────────────────────────────────────────
# Both fold the MaskedAddProjection target-replace (masked entries -> the
# finite pre-mask logit, i.e. zero pseudo-grad) into the softmax backward, so
# no -inf tensor is ever materialized in backward. `vis` is the boolean
# visibility mask; `qk` is the finite PRE-mask logits (equal to the masked
# logits at visible entries, since the mask adds exactly 0.0 there).

@torch.compile(dynamic=_TC_DYNAMIC)
def _softmax_mask_jacobian_impl(qk, z, p, vis):
    # Legacy SoftmaxProjection "jacobian": exact softmax VJP on the residual
    # s = p - z; masked entries have p = 0 and are excluded from s by `vis`.
    s = (p - z) * vis
    s_logit = p * (s - (p * s).sum(dim=-1, keepdim=True))
    return torch.where(vis, qk - s_logit, qk)


@torch.compile(dynamic=_TC_DYNAMIC)
def _softmax_mask_relfloor_impl(qk, z, p, vis, eps_rel: float, kappa: float,
                                clip: float, eps: float):
    # Legacy _softmax_relfloor_impl on (qk, vis) coordinates: identical values
    # at visible entries (safe_x == qk there); masked terms are zeroed by the
    # same `* vis` factors.
    z_pos = torch.maximum(z, eps_rel * p) * vis
    z_bar = z_pos / z_pos.sum(dim=-1, keepdim=True).clamp_min(eps)
    log_z = torch.log(z_bar.clamp_min(eps))
    diff = (qk - log_z) * vis
    c = diff.sum(dim=-1, keepdim=True) / vis.sum(dim=-1, keepdim=True).clamp(min=1)
    step = torch.clamp(log_z + c - qk, min=-clip, max=clip) * vis
    return torch.where(vis, qk + kappa * step, qk)
