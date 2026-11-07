"""Hopfield-retrieval backward for causal self-attention (ATTN_HOPFIELD=1).

Modern-Hopfield view (Ramsauer et al.; ml-jku.github.io/hopfield-layers):
attention IS one CCCP retrieval step of E(ξ) = −lse(β, Xᵀξ) + ½ξᵀξ —
per row, query q_i retrieves from a memory with addresses K and contents V:
p_i = softmax(q_i Kᵀ/ω), y_i = Vᵀ p_i.

The backward is therefore MEMORY EDITING, not message-passing through the
matmul→softmax→matmul composition (whose per-node solves manufacture the
1/N-diluted, rider-carrying, standing-demand targets — see
depth-ladder-parity notes):

  1. Retrieval retarget = Bregman (KL) projection in distribution space,
     closed form:  p_t ∝ p · exp(m/τ),  m_ij = v_j·δ_i,  δ = y_target − y.
     τ from the KL trust region via KL(p_t‖p) ≈ Var_p(m)/(2τ²)
     ⇒ τ = sqrt(Var_p(m) / (2·attn_hopfield_eps_kl)); tau scope "global"
     preserves backprop's row weighting (exact-VJP small-window limit,
     prototype cos 0.97), "row" gives per-row equal budget (flattens rows).
  2. Hebbian storage targets (all derived from the ONE p_t, idempotent at
     δ=0 — no solver internals to regenerate a standing demand):
       q_t = q + (p_t−p)@K/ω      (move probe toward addresses it should hit)
       k_t = k + (p_t−p)ᵀ@Q/ω     (strengthen associations that should fire)
       v_t = v + Pᵀδ              (content edit — the already-healthy adjoint)
     Each delta is self-parallel-deflated and capped into the td_eps_lin
     window per row (the attn_value_adjoint recipe).

Forward is numerically identical to the legacy composition (eager softmax
attention with the same mask/GQA-expand), so flag-off runs are bit-identical
and flag-on runs change credit assignment only, never the model.

Plugs in where FusedCausalAttentionProjection does: q (B,H,T,hd) post
norm/rotary, k,v (B,KV,T,hd) pre GQA-expand; y (B,H,T,hd) out; the
transpose/reshape/out-proj stay outside.

STATUS (2026-07-12, STATUS_LEDGER.md): eps_kl=1e-5 CONFIRMED optimal for the
ONE-SHOT solve — loosening the KL trust region monotonically hurts (3.4409 ->
3.835 across 1e-5 -> 1e-1; 1e-6 tie): the tight-tau direction is already
right, larger one-shot steps leave the softmax linear regime (the eps->0
limit of p_t - p is exactly the softmax VJP direction in the simplex
geometry). ATTN_HOPFIELD_ITERS=K iterates the prox (eps_kl/K per step, delta
recomputed) to consume a larger total budget without leaving the linear
regime per step. The former transit_vjp exact-adjoint branch was REMOVED
2026-07-12 (VJP banned; channel neutral in isolation: TRANSIT_VJP_ATTN=1
alone 3.4388 vs ctrl 3.4448). Backward math is torch.compile'd
(_hopfield_backward_impl, pure-tensor; config reads stay in the wrapper).
"""

import torch
import torch.nn.functional as F

from ..config import config
from .attention_fused import _TC_DYNAMIC, _causal_masks
from .ops import _deflate_cap, td_forward


def _expand(t, group):
    """GQA expand (B,KV,T,hd) -> (B,KV*group,T,hd), the legacy expand/reshape."""
    B, KV, T, hd = t.shape
    return t.unsqueeze(2).expand(B, KV, group, T, hd).reshape(B, KV * group, T, hd)


def _fold(d, group):
    """Sum a (B,H,T,hd) message over each KV head's query group."""
    return d.unflatten(1, (-1, group)).sum(2)


@torch.compile(dynamic=_TC_DYNAMIC)
def _hopfield_forward_impl(q, k, v, neg_inf, omega: float, group: int):
    """Compiled forward math (mask built outside; numerically identical to
    the eager composition — same expand/matmul/softmax order). Returns the row
    lse, not p: the backward recomputes p = exp(z − lse) from q, k."""
    qk = (q @ _expand(k, group).transpose(-2, -1)) / omega + neg_inf
    y = F.softmax(qk, dim=-1) @ _expand(v, group)
    return torch.logsumexp(qk, dim=-1, keepdim=True), y


@torch.compile(dynamic=_TC_DYNAMIC)
def _hopfield_backward_impl(q, k, v, neg_inf, lse, y, y_target, omega: float,
                            group: int, iters: int, eps_step: float,
                            tau_global: bool, td_eps_lin: float,
                            qk_amp: bool = False):
    """Pure-tensor backward math (config reads stay in the wrapper, repo
    convention for compiled backward impls). iters/tau_global/eps floats are
    compile-time constants — the inner KL-prox loop unrolls. Each prox step
    p_t ∝ p_t·exp(m/τ) is a shift of the logits z (exact, no log of p)."""
    k_exp, v_exp = _expand(k, group), _expand(v, group)
    z = (q @ k_exp.transpose(-2, -1)) / omega + neg_inf
    p = torch.exp(z - lse)
    delta = y_target - y
    p_t = p
    d_cur = delta
    for _ in range(iters):
        m = d_cur @ v_exp.transpose(-2, -1)               # (B,H,T,T)
        mean = (p_t * m).sum(-1, keepdim=True)
        var = (p_t * (m - mean).square()).sum(-1, keepdim=True)
        if tau_global:
            var = var.mean(dim=(-2, -1), keepdim=True)
        tau = (var / (2.0 * eps_step)).sqrt().clamp_min(1e-12)
        z = z + m / tau
        p_t = F.softmax(z, dim=-1)
        if iters > 1:
            d_cur = y_target - p_t @ v_exp
    dP = p_t - p

    # qk_amp (ATTN_HOPFIELD_QK_AMP): the KL trust-region linearizes
    # dP ~= (1/tau) * softmax_jac(m), so the q/k messages (built from dP) are
    # shrunk by 1/tau relative to the value message dv (built directly from
    # delta, no tau). Measured (probe_softmax_transport.py): this INVERTS the
    # gradient's q/k-vs-v amplitude ratio (true |dq|/|dk| >> |dv|; hopfield gets
    # |dv| >> |dq|), dropping the joint q/k/v alignment 0.74 -> 0.24. Restoring
    # the tau factor on dq/dk recovers the exact-VJP alignment (0.7373 vs
    # 0.7372). A positive per-message scalar = own-op / PJAX-legal (no covector
    # chain). tau is (B,H,1,1) global or (B,H,T,1) per query row: it scales dP's
    # rows, so dk takes it on q before the dPᵀ contraction over query rows.
    qk_scale = tau if qk_amp else 1.0
    dq = dP @ k_exp / omega * qk_scale
    dk = _fold(dP.transpose(-2, -1) @ (q * qk_scale) / omega, group)
    dv = _fold(p.transpose(-2, -1) @ delta, group)
    return (q + _deflate_cap(dq, q, td_eps_lin), k + _deflate_cap(dk, k, td_eps_lin),
            v + _deflate_cap(dv, v, td_eps_lin))


# ---------------------------------------------------------------------------
# alpha-entmax attention (config.attn_entmax, 2026-07-13). Forward = alpha-entmax
# (Peters/Martins 2019) instead of softmax; backward = the Tsallis-mirror prox
# (entmax's Bregman analog of the softmax KL-prox). entmax zeros the ~1024-key
# tail that gives the softmax Jacobian condition ~1e3-1e5, so the projection's
# 1/sigma transport lands ~350-890x better conditioned (probed on real patterns).
# Gated so flag-off never runs here. alpha=1 -> softmax, alpha=2 -> sparsemax;
# validated best at ~1.5 with support ~25-50 (use a wider attention scale /
# temperature so support does not collapse to <5, where the prox stops
# recovering the VJP).
# ---------------------------------------------------------------------------
def _entmax_masked(z, alpha, tau=None, n_iter=40):
    """alpha-entmax over the last dim. Masked positions carry z=-inf and map to
    p=0. p_i = relu((alpha-1) z_i - tau)^(1/(alpha-1)), tau by bisection unless
    given (the backward's recompute from the saved tau). Returns (p, tau)."""
    am1 = alpha - 1.0
    p_exp = 1.0 / am1
    s = am1 * z
    if tau is None:
        s_finite = torch.where(torch.isfinite(s), s, torch.full_like(s, -1e30))
        tau_hi = s_finite.max(dim=-1, keepdim=True).values
        tau_lo = tau_hi - 10.0
        for _ in range(n_iter):
            tau = 0.5 * (tau_lo + tau_hi)
            f = torch.clamp(s - tau, min=0.0).pow(p_exp).sum(-1, keepdim=True) - 1.0
            too_big = f > 0
            tau_lo = torch.where(too_big, tau, tau_lo)
            tau_hi = torch.where(too_big, tau_hi, tau)
    p = torch.clamp(s - tau, min=0.0).pow(p_exp)
    return p / p.sum(-1, keepdim=True).clamp_min(1e-30), tau


class Entmax(torch.autograd.Function):
    """Gradient-mode alpha-entmax over the last dim (z=-inf masked). Autograd
    through the bisection is wrong; the closed-form VJP J g = w⊙g − w(wᵀg)/Σw,
    w = p^(2−α) on the support, needs only p, so p is all it saves."""

    @staticmethod
    def forward(ctx, z, alpha):
        p, _ = _entmax_masked(z, alpha)
        ctx.save_for_backward(p)
        ctx.alpha = alpha
        return p

    @staticmethod
    def backward(ctx, g):
        p, = ctx.saved_tensors
        w = torch.where(p > 0, p.pow(2.0 - ctx.alpha), 0.0)
        wg = w * g
        return wg.sub_(w.mul_(wg.sum(-1, keepdim=True) / w.sum(-1, keepdim=True))), None


@torch.compile(dynamic=_TC_DYNAMIC)
def _entmax_forward_impl(q, k, v, neg_inf, omega, group, alpha):
    """Returns the row threshold tau (not p) and y."""
    p, tau = _entmax_masked((q @ _expand(k, group).transpose(-2, -1)) / omega + neg_inf, alpha)
    return tau, p @ _expand(v, group)


@torch.compile(dynamic=_TC_DYNAMIC)
def _entmax_backward_impl(q, k, v, neg_inf, tau_p, y, y_target, omega, group,
                          eps, tau_global, td_eps_lin, alpha):
    """Tsallis-mirror retrieval retarget, the exact threshold-shift step
    p_t = entmax(p^(a-1)/(a-1) + m/tau) (mirror map of the Tsallis entropy;
    reduces to the softmax p*exp(m/tau) as a->1), solved by the forward's own
    threshold bisection on the forward's -inf mask: invariant to row shifts of
    m (value offsets), and p_t is a distribution on the visible keys. dv is the
    exact value adjoint P^T delta, same as softmax. Measured vs the true entmax
    JVP (tau global, eps 1e-5, alpha 1.5, support 2-10 of 24): dP cos
    0.997-1.000 at value offsets 0/1/3 sigma; the old division-renormalized
    step gave 0.97/0.87/0.59 (support 5)."""
    k_exp, v_exp = _expand(k, group), _expand(v, group)
    am1 = alpha - 1.0
    p, _ = _entmax_masked((q @ k_exp.transpose(-2, -1)) / omega + neg_inf, alpha, tau_p)
    delta = y_target - y
    m = delta @ v_exp.transpose(-2, -1)                    # (B,H,T,T)
    mean = (p * m).sum(-1, keepdim=True)
    var = (p * (m - mean).square()).sum(-1, keepdim=True)
    if tau_global:
        var = var.mean(dim=(-2, -1), keepdim=True)
    tau = (var / (2.0 * eps)).sqrt().clamp_min(1e-12)
    p_t, _ = _entmax_masked(p.pow(am1) / am1 + neg_inf + m / tau, alpha)
    dP = p_t - p
    dq = dP @ k_exp / omega
    dk = _fold(dP.transpose(-2, -1) @ q / omega, group)
    dv = _fold(p.transpose(-2, -1) @ delta, group)
    return (q + _deflate_cap(dq, q, td_eps_lin), k + _deflate_cap(dk, k, td_eps_lin),
            v + _deflate_cap(dv, v, td_eps_lin))


# ---------------------------------------------------------------------------
# ReLU attention (config.attn_relu, 2026-07-13). a = relu(qk^T/scale) / n with
# n = causal row length (fixed). The derivative da/dlogit is a diagonal 0/1 gate
# -> condition EXACTLY 1 on the active set, and preimage == adjoint (cos 1.0,
# err 0 on real patterns), so the projection's 1/sigma transport becomes EXACT
# through attention (the whole softmax disease removed structurally). Keeps a
# dense active set (~40-50% of keys), unlike sparsemax. Backward = the exact
# local adjoint (== finite differences), deflated+capped like the other nodes.
# a/n chosen over renorm (renorm re-couples the Jacobian) and /n_active
# (L-dependent -> inexact backward). Own-op, PJAX-legal. Default off.
# ---------------------------------------------------------------------------
class ReLUAttentionProjection(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, omega):
        group = q.shape[1] // k.shape[1]
        _, visible = _causal_masks(q.shape[2], q.device, q.dtype)
        n = visible.sum(-1, keepdim=True).to(q.dtype).clamp_min(1.0)  # (T,1)
        logits = (q @ _expand(k, group).transpose(-2, -1)) / omega
        y = ((torch.relu(logits) * visible.to(q.dtype)) / n) @ _expand(v, group)
        ctx.omega = float(omega)
        ctx.group = group
        ctx.save_for_backward(q, k, v, y)      # logits recomputed in backward
        return y

    @staticmethod
    def backward(ctx, y_target):
        q, k, v, y = ctx.saved_tensors
        td_forward(ctx, q, k, v)
        group = ctx.group
        _, visible = _causal_masks(q.shape[2], q.device, q.dtype)
        n = visible.sum(-1, keepdim=True).to(q.dtype).clamp_min(1.0)
        k_exp, v_exp = _expand(k, group), _expand(v, group)
        logits = (q @ k_exp.transpose(-2, -1)) / ctx.omega
        gate = (logits > 0) & visible                           # active set
        a = logits.mul_(gate).div_(n)
        delta = y_target - y
        # exact local adjoint (== preimage, diagonal 0/1 gate): move q,k,v so the
        # output moves toward delta. dv = a^T delta; dlogit = (delta V^T ⊙ gate)/n.
        dv = _fold(a.transpose(-2, -1) @ delta, group)
        dlogit = (delta @ v_exp.transpose(-2, -1)).mul_(gate).div_(n)  # (B,H,T,T)
        dq = dlogit @ k_exp / ctx.omega
        dk = _fold(dlogit.transpose(-2, -1) @ q / ctx.omega, group)
        td_eps_lin = float(config.td_eps_lin)
        return (q + _deflate_cap(dq, q, td_eps_lin), k + _deflate_cap(dk, k, td_eps_lin),
                v + _deflate_cap(dv, v, td_eps_lin), None)


class HopfieldAttentionProjection(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, omega):
        group = q.shape[1] // k.shape[1]
        neg_inf, _ = _causal_masks(q.shape[2], q.device, q.dtype)
        alpha = float(config.attn_entmax_alpha)
        # alpha = 1 is the softmax limit of entmax (1/(alpha-1) is singular
        # there): run the plain softmax Hopfield node, which IS that limit.
        ctx.entmax = config.attn_entmax and alpha > 1.0
        ctx.alpha = alpha
        if ctx.entmax:
            stat, y = _entmax_forward_impl(q, k, v, neg_inf, float(omega), group, alpha)
        else:
            stat, y = _hopfield_forward_impl(q, k, v, neg_inf, float(omega), group)
        ctx.omega = float(omega)
        ctx.group = group
        # Row statistic (lse, or the entmax threshold) instead of p: the
        # backward recomputes p from q, k.
        ctx.save_for_backward(q, k, v, stat, y)
        return y

    @staticmethod
    def backward(ctx, y_target):
        # KL-prox retrieval retarget (closed form), optionally iterated:
        # K inner steps of budget eps_kl/K each, recomputing delta from the
        # updated retrieval between steps (y_cur = p_cur @ V). One-shot (K=1,
        # default, bit-identical) linearizes m at the ORIGINAL y, which is why
        # loosening eps_kl monotonically hurts; iterating keeps each step in
        # the linear regime while consuming a K-times larger total trust
        # region — more standing-demand throughput per training step.
        # Then Hebbian storage targets, one consistent allocation from p_t.
        # (The former transit_vjp exact-adjoint branch was REMOVED 2026-07-12:
        # VJP is banned in this program, and the channel measured neutral in
        # isolation — TRANSIT_VJP_ATTN=1 alone 3.4388 vs ctrl 3.4448.)
        q, k, v, stat, y = ctx.saved_tensors
        td_forward(ctx, q, k, v)
        neg_inf, _ = _causal_masks(q.shape[2], q.device, q.dtype)
        iters = max(config.attn_hopfield_iters, 1)
        eps_kl = config.attn_hopfield_eps_kl
        tau_global = config.attn_hopfield_tau == "global"
        if ctx.entmax:
            q_t, k_t, v_t = _entmax_backward_impl(
                q, k, v, neg_inf, stat, y, y_target, ctx.omega, ctx.group, eps_kl,
                tau_global, float(config.td_eps_lin), ctx.alpha)
        else:
            q_t, k_t, v_t = _hopfield_backward_impl(
                q, k, v, neg_inf, stat, y, y_target, ctx.omega, ctx.group, iters,
                eps_kl / iters, tau_global, float(config.td_eps_lin),
                config.attn_hopfield_qk_amp)
        return q_t, k_t, v_t, None
