"""Experimental modules restored from the openAI_challenge work.

Originally lived in `experimental_modules.py` (deleted in commit fd03390a2);
this file pulls back the framework-aware multi-head attention layer
associated with the openAI_challenge experiment, switched to use the new
`SoftmaxProjection` (previously `SimplexProjection`).

Caveat: reproduced from the deleted source as a starting point — not
re-verified end-to-end.
"""

import itertools
import os
from typing import Optional
from ptorch.core.ops import MaskedAddProjection, mm_prox_site, td_sibling_group

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import config
from ..core.ops import MatMulProjection, SoftmaxProjection, RMSNormProjection, Conversion as ConversionFn, EmbeddingProjection, SeqMaxPoolProjection, SeqAvgPoolProjection, BranchProjection, HardtanhProjection, HadamardProjection, td_repeat_target, td_forward
from ..core.attention_fused import _causal_masks, fused_attention_enabled
from ..core.attention_native import FusedCausalAttentionProjectionV2, Sparsemax, SparsemaxProjection
from ..core.dedilute import dedilute
from ..core.delta_probe import probe
from ..core.attention_hopfield import HopfieldAttentionProjection, ReLUAttentionProjection, Entmax
from .modules import Linear, ProjectionModule

# PTORCH_COMPILE_DYNAMIC: 1 (default, symbolic shapes) / 0 (static-shape
# specialization; faster kernels when shapes are constant, recompiles per
# distinct shape) / auto (torch default).
_TC_DYNAMIC = {"1": True, "0": False, "auto": None}[os.environ.get("PTORCH_COMPILE_DYNAMIC", "1")]


class ResidualAdd(torch.autograd.Function):
    """Projection-aware addition for residual connections (canonical).

    Consolidates the three formerly-duplicated experiment-local copies
    (ViT / GPT / CNN benchmarks). ``mode="full"`` routes the FULL displacement
    ``delta = z_target - (skip + out)`` to both addends (the backprop-residual
    gain-1 analog / "complementary" split target, so skip carries the signal
    down the backbone with gain 1); ``mode="half"`` restores the strict
    min-norm split (``delta / 2`` each).

    TD graph scope only: forwards the incoming per-edge floor to *both* inputs'
    producers (this sits on every trunk edge — a registry miss here would sever
    the trace upstream). In global scope it is TD-agnostic (unchanged, so the
    consolidation is behavior-preserving for existing runs).

    NOTE (2026-07-12): equal-weight fan-in routing is path-scalar INCONSISTENT
    under displacement transport (the sublayer path arrives with a different
    accumulated scalar than the skip — off by rms^2-type factors per crossed
    norm); the fix is trainer-level MERGE_RHO fan-in equalization, not a
    change here — see transport_theory.md T2 (no-VJP record 3.2465).
    """
    @staticmethod
    def forward(ctx, skip, out, mode="full"):
        ctx.save_for_backward(*((skip, out) if config.use_projections else ()))
        ctx.mode = mode
        return skip + out

    @staticmethod
    def backward(ctx, z_target):
        if not config.use_projections:
            return z_target, z_target, None
        skip, out = ctx.saved_tensors
        delta = z_target - (skip + out)
        if ctx.mode == "half":
            delta = delta / 2.0
        td_forward(ctx)
        return skip + delta, out + delta, None

class GQAConsensusProjection(torch.autograd.Function):
    """
    Forward: expands the KV tensor (bsz, kv_heads, T, hd) to the query heads'
    (bsz, kv_heads, group, T, hd). Backward: the group's targets combine like
    a Branch fan-out (config.branch_mode): "mean" averages them, "delta_sum"
    returns x + sum_g (t_g - x).
    """
    @staticmethod
    def forward(ctx, x, group_size):
        ctx.mode = config.branch_mode
        ctx.save_for_backward(x)
        return x.unsqueeze(2).expand(-1, -1, group_size, -1, -1)

    @staticmethod
    def backward(ctx, z_target):
        (x,) = ctx.saved_tensors
        td_forward(ctx)
        if ctx.mode == "delta_sum":
            return z_target.sum(dim=2) - (z_target.size(2) - 1) * x, None
        return z_target.mean(dim=2), None


def gqa_expand(x, group):
    """(bsz, kv_heads, T, hd) -> (bsz, kv_heads*group, T, hd); projection mode
    combines the group's targets in GQAConsensusProjection."""
    if group > 1:
        x = (GQAConsensusProjection.apply(x, group) if config.use_projections
             else x.unsqueeze(2).expand(-1, -1, group, -1, -1)).flatten(1, 2)
    return x


class Branch(nn.Module):
    """Returns ``num_branches`` references of the input. In projection mode each
    reference is wrapped with its own ``BranchProjection.apply(x, N, mode)``.
    With ``mode="mean"`` (legacy default) the target flowing back to ``x`` is
    the *average* of the per-branch targets; with ``mode="delta_sum"`` it is
    ``x + sum_i (t_i - x)``, the backprop fan-out analog that preserves gain-1
    skip paths in residual nets. ``mode=None`` defers to ``config.branch_mode``
    at call time. In non-projection mode this is a plain identity tuple
    (standard summing preserved).

    Returning independent references (rather than a single shared one) keeps
    ``MultiheadAttention``'s ``q is k and k is v`` self-attention check inert,
    avoiding a second BranchProjection wrap inside the attention module.

    NOTE (2026-07-12): neither combine mode can repair fan-in messages whose
    accumulated path scalars disagree (metric-blind; probe P3 merged cos
    0.044 for both) — the working fix is trainer-level MERGE_RHO fan-in
    equalization, transport_theory.md T2.
    """
    def __init__(self, num_branches: int, mode: Optional[str] = None):
        super().__init__()
        self.num_branches = int(num_branches)
        self.mode = mode

    def forward(self, x: torch.Tensor):
        if config.use_projections:
            mode = self.mode if self.mode is not None else config.branch_mode
            return tuple(BranchProjection.apply(x, self.num_branches, mode)
                         for _ in range(self.num_branches))
        return tuple(x for _ in range(self.num_branches))


class SeqMaxPool(nn.Module):
    """GNN-style global max-pool readout over the sequence/token dimension.

    Drop-in replacement for CLS-token aggregation: maps (B, T, D) -> (B, D)
    by taking the per-feature max across tokens. Under projection mode,
    backward solves the closed-form max-projection per (batch, channel).
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if config.use_projections:
            return SeqMaxPoolProjection.apply(x)
        return torch.amax(x, dim=1)


class SeqAvgPool(nn.Module):
    """GNN-style global average-pool readout over the sequence/token dimension.

    Drop-in replacement for CLS-token aggregation: maps (B, T, D) -> (B, D)
    by taking the per-feature mean across tokens. Under projection mode,
    backward solves the closed-form mean-projection per (batch, channel).
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if config.use_projections:
            return SeqAvgPoolProjection.apply(x)
        return x.mean(dim=1)


class RMSNorm(ProjectionModule):
    def __init__(self, eps: float = 1e-5):
        super().__init__(1)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if config.use_projections:
            return RMSNormProjection.apply(x, self.eps)
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class Scale(ProjectionModule):
    """Learnable per-dim element-wise scale: y = weight * x.

    Uses HadamardProjection: x receives its projection target and weight.grad
    the residual weight - target, like every parameter → use ProjectionAdam.
    init: initial value for all weight elements (1.0 for scale, small nonzero
    for additive blending to avoid division-by-zero in the backward).
    """
    def __init__(self, dim: int, init: float = 1.0):
        super().__init__(1)
        self.weight = nn.Parameter(torch.full((dim,), float(init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if config.use_projections:
            return HadamardProjection.apply(self.weight, x)
        return self.weight * x


class DyT(ProjectionModule):
    """Normalization-free DyT-style layer: y = clamp(alpha * x, -1, 1).

    Drop-in alternative to RMSNorm whose backward is an *exact* closed-form
    graph projection (HardtanhProjection) rather than a renormalizing
    reconstruction — it preserves per-token magnitude information and cannot
    amplify vanished targets. alpha is a fixed gain (learnable alpha would
    need its own bilinear projection treatment).
    """
    def __init__(self, alpha: float = 0.5):
        super().__init__(1)
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if config.use_projections:
            return HardtanhProjection.apply(x, self.alpha)
        return torch.clamp(self.alpha * x, -1.0, 1.0)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies the rotary embedding to the input tensor."""
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


@torch.compile(dynamic=_TC_DYNAMIC)
def _rotary_midpoint_impl(x, z_target, cos, sin):
    # Inverse rotation (negated sine) of the target, then the exact L2
    # midpoint projection onto z = R x.
    rt_z_target = apply_rotary_emb(z_target, cos, -sin)
    return (x + rt_z_target) / 2.0

class RotaryFunction(torch.autograd.Function):
    """
    Custom PTorch Autograd Function for Rotary Embeddings.
    Instead of calculating gradients, the backward pass computes the L2 projection
    onto the RoPE architectural constraint set.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x, cos, sin)
        return apply_rotary_emb(x, cos, sin)

    @staticmethod
    def backward(ctx, z_target: torch.Tensor):
        """
        dz is the residual passed from the layer above (z - z_target).
        We use it to reconstruct the target, compute the projection, and 
        return our own residual to pass to the layer below.
        """
        x, cos, sin = ctx.saved_tensors

        # Exact L2 projection onto the constraint z = R x: midpoint between the
        # forward input and the inversely-rotated target output (fused impl).
        x_bar = _rotary_midpoint_impl(x, z_target, cos, sin)

        # Direction-pipe repeater: the midpoint halves the residual like the
        # active ReLU branch; restore its amplitude (no-op at td_lambda=0).
        return td_repeat_target(x, x_bar, ctx), None, None


class Rotary(nn.Module):
    """
    PTorch-compliant Rotary Positional Embedding layer.
    """
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            # Build outside inference mode: the cache may be created during an
            # inference_mode eval and later reused by a training forward, where
            # inference tensors cannot be saved for backward.
            with torch.inference_mode(False):
                t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
                freqs = torch.outer(t, self.inv_freq.to(device))
                self._cos_cached = freqs.cos()[None, None, :, :]
                self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len

    def forward(self, x: torch.Tensor, seq_len: int = None) -> torch.Tensor:
        """
        Applies PTorch-compliant RoPE to the input tensor `x`.
        Expects x shape to be roughly (batch, heads, seq_len, head_dim).
        """
        if seq_len is None:
            seq_len = x.size(-2)

        self._update_cache(seq_len, x.device, x.dtype)
        cos = self._cos_cached.to(dtype=x.dtype)
        sin = self._sin_cached.to(dtype=x.dtype)

        if config.use_projections:
            return RotaryFunction.apply(x, cos, sin)
        return apply_rotary_emb(x, cos, sin)

class Softcap(nn.Module):
    """Tanh softcap y = C * tanh(x / C) — GRADIENT PATH ONLY.

    DO NOT USE FOR LOGIT SOFTCAP UNDER use_projections. The projection
    backward this module used to dispatch to (LogitSoftcapInversion, the
    exact atanh inversion of the cap) was removed in the 2026-07-09 cleanup
    after diagI1 diverged at full size: the inverse has local gain
    1/(1 - (z/C)^2) -> inf at the cap, so it AMPLIFIES targets near
    saturation exactly where the gradient backward (sech^2 -> 0) damps them,
    and its +-(C - eps) clamp teleports saturated targets to
    C * atanh(1 - eps/C) ~ 9*C in one step — a positive-feedback loop with
    weight inflation (logit_max pinned at the cap, val 4.2 -> 5.1; see
    experiments/diagnose/FINDINGS_scale_dive.md). Never reintroduce the
    exact inversion. A projection-safe cap would have to be the graph
    projection of y = C * tanh(x / C) (which moves the OUTPUT target back
    toward feasible near saturation, i.e. damps); no such card exists, so
    the projection path raises instead of silently misbehaving.
    """

    def __init__(self, logit_softcap=30.0):
        super().__init__()
        self.logit_softcap = logit_softcap

    def forward(self, x):
        if config.use_projections:
            raise NotImplementedError(
                "Softcap has no projection backward: LogitSoftcapInversion "
                "(exact atanh inverse) was removed after diagI1 diverged — "
                "it amplifies targets near the cap instead of damping. Do "
                "not use Softcap for logit softcap under projections; see "
                "the class docstring / FINDINGS_scale_dive.md."
            )
        return self.logit_softcap * torch.tanh(x / self.logit_softcap)

_layer_counter = itertools.count()


class _StopDemand(torch.autograd.Function):
    """Identity forward; backward returns the saved input as the target
    (delta = 0), cutting the demand transported down this branch. Diagnostic
    knob for the depth-ladder attention-transit work (config.attn_xqk_stop stops
    the q/k branch demand into x); projection mode only."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, target):
        (x,) = ctx.saved_tensors
        return x.detach()


class MultiheadAttention(nn.Module):
    """Projection-aware general multi-head attention with RoPE + GQA + RMSNorm:
    self- or cross-attention via separate q, k, v inputs, an additive
    ``attn_mask`` or ``is_causal``. The one attention dispatcher (both modes):

      attn_relu > attn_hopfield (entmax) > fused node (fused_attn, inside
      fused_attention_enabled) > legacy node composition (softmax/sparsemax
      per attention_activation).

    relu/hopfield/fused are causal-only nodes, so they apply when is_causal
    and no attn_mask is given; gradient mode computes the same forward with
    plain differentiable torch. qk_gamma, attn_temp, attn_xqk_stop,
    mm_prox_site("attn"), the q/k/v TD sibling group and dedilute apply to
    every call. δ probes (core/delta_probe.py) mark the segment boundaries
    (inert unless the RECORDER is configured).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        norm: str = "l2",
        qk_match_g_to_omega: bool = False,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        # With omega=sqrt(head_dim) and g=1, matmul_proj's target_penalty
        # (omega/g)^2 = head_dim makes the QK^T solve dump the constraint gap
        # into the *discarded* Z_proj instead of moving q,k. Matching g to
        # omega restores target_penalty = 1 so the scale factor stops
        # weakening the learning signal into the q/k projections.
        self.qk_match_g_to_omega = qk_match_g_to_omega

        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        kv_dim = self.num_kv_heads * self.head_dim

        self.c_q = Linear(dim, dim, bias=False, norm=norm)
        self.c_k = Linear(dim, kv_dim, bias=False, norm=norm)
        self.c_v = Linear(dim, kv_dim, bias=False, norm=norm)
        self.proj = Linear(dim, dim, bias=False, norm=norm)

        self.rotary = Rotary(self.head_dim, base=rope_base)
        self.q_norm = RMSNorm()
        self.k_norm = RMSNorm()
        self._td_sibling = {"size": 3}
        # Depth tag for δ records ("attn{L}.…"), in construction order.
        self.delta_tag = next(_layer_counter)

    def _project_pairwise_matmul(self, left: torch.Tensor, right: torch.Tensor,
                                  omega: float = 1.0,
                                  g: Optional[float] = None,
                                  value_solve: bool = False) -> torch.Tensor:
        if not config.use_projections:
            return (left @ right) / omega
        return MatMulProjection.apply(
            left, right, config.attn_pairwise_num_steps,
            config.projection_alpha,
            g if g is not None else config.projection_g,
            omega, True, value_solve,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        bsz, seqlen, dim = q.shape
        seqlen_k = k.shape[1]
        tag = self.delta_tag
        proj = config.use_projections

        if proj:
            probe(q, f"attn{tag}.x")
            # A tensor feeding several of q/k/v gets one BranchProjection node
            # per consumer (Branch semantics, config.branch_mode): autograd
            # would otherwise sum their absolute targets.
            ins = (q, k, v)
            q, k, v = [BranchProjection.apply(t, n, config.branch_mode)
                       if (n := sum(t is u for u in ins)) > 1 else t for t in ins]
            if config.attn_xqk_stop:
                q, k = _StopDemand.apply(q), _StopDemand.apply(k)

        # mm_prox_site: tag the attention Linear cards (c_q/c_k/c_v and proj
        # below) so config.mm_prox_scope="attn" can target the cond-1e6 hops
        # without touching the healthy MLP/head cards. No-op unless
        # config.mm_prox_mu > 0.
        with mm_prox_site("attn"), td_sibling_group(self._td_sibling):
            q = self.c_q(q).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.c_k(k).reshape(bsz, seqlen_k, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = self.c_v(v).reshape(bsz, seqlen_k, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rotary(self.q_norm(q), seqlen)
        k = self.rotary(self.k_norm(k), seqlen_k)

        scale = (self.head_dim ** 0.5) * config.attn_temp
        g_qk = config.qk_gamma * (scale if self.qk_match_g_to_omega else config.projection_g)
        group = self.num_heads // self.num_kv_heads
        causal = is_causal and attn_mask is None
        relu = causal and config.attn_relu
        hop = causal and not relu and config.attn_hopfield
        if proj and relu:
            # ReLU attention: a=relu(qk^T/scale)/n, exact-transport (diagonal
            # 0/1 gate). GQA handled inside the node.
            y = ReLUAttentionProjection.apply(q, k, v, scale)
        elif proj and hop:
            # Attention as Hopfield retrieval: backward is memory editing, see
            # ptorch/core/attention_hopfield.py.
            y = HopfieldAttentionProjection.apply(q, k, v, scale)
        elif causal and fused_attention_enabled():
            # One autograd node for the whole core; saves q,k,v + a row
            # statistic instead of the per-layer (B,H,T,T) chain.
            y = FusedCausalAttentionProjectionV2.apply(q, k, v, scale, g_qk, tag)
        else:
            if causal:
                attn_mask = _causal_masks(seqlen, q.device, q.dtype)[0]
            qk = self._project_pairwise_matmul(q, gqa_expand(k, group).transpose(-2, -1),
                                               omega=scale, g=g_qk)
            probe(qk, f"attn{tag}.qk")
            if attn_mask is not None:
                qk = MaskedAddProjection.apply(qk, attn_mask) if proj else qk + attn_mask
            sparse = config.attention_activation == "sparsemax" and not hop
            if proj:
                attention_weights = (SparsemaxProjection if sparse else SoftmaxProjection).apply(qk)
            elif relu:
                attention_weights = torch.relu(qk) / _causal_masks(seqlen, q.device, q.dtype)[1].sum(-1, keepdim=True)
            elif hop and config.attn_entmax and config.attn_entmax_alpha > 1.0:
                attention_weights = Entmax.apply(qk, float(config.attn_entmax_alpha))
            elif sparse:
                attention_weights = Sparsemax.apply(qk, qk != float("-inf"))
            else:
                attention_weights = F.softmax(qk, dim=-1)
            probe(attention_weights, f"attn{tag}.pattern")
            y = self._project_pairwise_matmul(attention_weights, gqa_expand(v, group), value_solve=True)
        probe(y, f"attn{tag}.pv")
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        with mm_prox_site("attn"):
            out = self.proj(dedilute(y, self.proj.out_features))
        if proj:
            probe(out, f"attn{tag}.out")
        return out


class CausalSelfAttention(MultiheadAttention):
    """Projection-aware causal self-attention: MultiheadAttention(x, x, x,
    is_causal=True)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x, x, x, is_causal=True)


class Embedding(ProjectionModule):
    """Projection-based token embedding (row-gather via EmbeddingProjection).

    Drop-in replacement for the ``nn.Embedding`` + ``Conversion`` gradient
    bridge: the weight receives a scatter-mean projection residual (not a real
    gradient) in its ``.grad`` and is updated by the Projection* optimizers like
    every other projection parameter — no Conversion, no separate Adam.

    Weight tying: point an lm_head ``Linear.weight`` at ``self.weight`` (same
    Parameter, both shape ``(vocab, dim)``), and ``.grad`` holds the sum of the
    two residuals. Their scales differ by ~1000x, so to weigh them in the
    training loop also set ``self.tied = True``: the embedding's residual then
    goes to ``self.tied_sink[0]`` and ``.grad`` holds the head's alone; see
    EmbeddingProjection and accumulate_step.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, init_std: float = 0.02):
        super().__init__(1)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.normal_(self.weight, mean=0.0, std=init_std)
        self.tied = False
        self.tied_sink = [None]

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if config.use_projections:
            sink = self.tied_sink if self.tied else None
            return EmbeddingProjection.apply(self.weight, ids, sink)
        return F.embedding(ids, self.weight)


class Conversion(nn.Module):
    """Bridge layer: converts projection targets into real gradients.
    
    Forward: identity (pass-through).
    Backward: receives projection target, returns gradient (input - target)
    so that upstream gradient-based layers (e.g. Embedding) can learn.
    """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        if config.use_projections:
            return ConversionFn.apply(input)
        return input
