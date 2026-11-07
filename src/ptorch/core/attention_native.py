"""Projection-native attention core.

  1. ``SparsemaxProjection`` — an exact projection node for the score→weights
     map. sparsemax(s) = Π_Δ(s) is itself the Euclidean projection onto the
     simplex; its graph is a union of affine faces indexed by support sets, so
     projecting onto the graph is the same branch-comparison game as the ReLU
     card — exact, not heuristic (unlike SoftmaxProjection's two-sided
     clamp/renormalize/log-shift).
  2. ``FusedCausalAttentionProjectionV2`` — the fused attention node (see
     core/attention_fused.py for its envelope and shared kernels), covering
     both activations (config.attention_activation ∈ {"softmax",
     "sparsemax"}) plus two validated transport fixes at the
     attention-internal cards: per-entry sum transport
     (config.attn_exact_transport — the regular per-entry bilinear
     projections WITHOUT the consensus average: the same matmul_proj Newton
     solve per product entry, copies recombined by summing their transported
     parts; fixes the fp32 precision collapse of the consensus min-norm scale;
     the only variant that learns the needle-retrieval bench) and the
     operand-parallel rider strip (config.attn_strip_self — zeroes the
     persistent radial DC that drives the w_max drift). Otherwise it
     reproduces the legacy composition's backward, including the
     attn_value/prob/qk_adjoint overrides and proj_out_cubic.
     Pruned after measurement 2026-07-09 (do not resurrect without new
     evidence): inner sweeps + Dykstra (no val gain, deepened the Scale
     dive), the consensus-free alternating-ridge Π_A solve (loss-blind
     preconditioner), and mean→sum recombination (provably pure scale).

All config keys used here (attention_activation, sparsemax_max_face_iters,
sparsemax_gamma, attn_exact_transport, attn_strip_self,
attn_transport_eps_scale, attn_pairwise_num_steps, p_saved, fused_attn) are
ordinary Config dataclass fields in config.py — covered by
``config.snapshot()``/``reset()`` like any other key; the per-key rationale
lives with the field definitions.

Notational conventions follow the existing projection cards: forward point
(h, θ, h⁺) resp. (s, a); incoming output target ā (aka z); α weights the
B-side move, γ = g the output-target stiffness with target_penalty = (ω/g)².

TD repeaters: the activation node's own repeater is skipped and the mask-node
repeater runs on the finite pre-mask logits (see attention_fused.py); the
sparsemax card follows the same convention so fused and legacy paths agree.
"""

import torch
import torch.nn.functional as F

from ..config import config
from . import ops
from .attention_fused import (_causal_masks, _softmax_mask_jacobian_impl,  # noqa: F401 (_causal_masks re-exported)
                              _softmax_mask_relfloor_impl)
from .ops import (TD_TRACE, _deflate_cap, matmul_proj, td_bilinear_floor_update,
                  td_forward, td_repeat_target)
from .delta_probe import RECORDER

def activation_gamma() -> float:
    """γ for the sparsemax activation card: sparsemax_gamma override if set,
    else the shared projection_g (bit-identical legacy behavior)."""
    if config.sparsemax_gamma is not None:
        return float(config.sparsemax_gamma)
    return float(config.projection_g)


# ── sparsemax forward ─────────────────────────────────────────────────────────

_NEG_FILL = -1e30  # finite stand-in for -inf inside sort/cumsum arithmetic


def sparsemax_forward(scores: torch.Tensor, vis: torch.Tensor):
    """Row-wise sparsemax over the last dim, restricted to the allowed set.

    ``a = Π_Δ(s)`` via the sort-based threshold (Held et al. 1974 / Martins &
    Astudillo 2016), the same algorithm as ops._project_simplex_sorted:
    sort masked scores descending, ρ = |{k : 1 + k·u_(k) > Σ_{j<=k} u_(j)}|,
    τ = (Σ_{k<=ρ} u_(k) − 1)/ρ, a_k = max(s_k − τ, 0). Disallowed entries are
    filled with a large negative finite value (never inf, so no NaN in the
    cumsum) and end up exactly 0.

    Returns ``(a, tau)`` with ``tau`` shaped (..., 1). The support is
    recoverable as ``(s − τ > 0) & vis`` (equivalently ``a > 0``), so only τ —
    one float per row, the sparsemax analog of ``lse`` — needs saving.
    """
    z = torch.where(vis, scores, _NEG_FILL)
    u = torch.sort(z, dim=-1, descending=True).values   # int64 indices freed here
    css = u.cumsum(-1).sub_(1.0)
    j = torch.arange(1, z.size(-1) + 1, device=z.device, dtype=z.dtype)
    # u > css/j  <=>  u - css/j > 0 exactly (IEEE subtraction of finite values)
    rho = (css / j).lt_(u).sum(dim=-1, keepdim=True).clamp_(min=1)
    tau = torch.gather(css, -1, rho.long() - 1) / rho
    return z.sub_(tau).clamp_(min=0.0), tau


class Sparsemax(torch.autograd.Function):
    """Gradient-mode ``sparsemax(scores)`` over the allowed set ``vis``. The
    closed-form VJP (g − mean_S g on the support S = {a > 0}, 0 off it) needs
    only the output a, so a is all it saves (autograd through the sort-based
    forward saves ~4 TT)."""

    @staticmethod
    def forward(ctx, scores, vis):
        a, _ = sparsemax_forward(scores, vis)
        ctx.save_for_backward(a)
        return a

    @staticmethod
    def backward(ctx, g):
        S = ctx.saved_tensors[0] > 0
        g = torch.where(S, g, 0.0)
        return g.sub_(g.sum(-1, keepdim=True) / S.sum(-1, keepdim=True)).mul_(S), None


# ── sparsemax graph projection (the exact backward) ───────────────────────────

@torch.compile(dynamic=ops._TC_DYNAMIC)
def sparsemax_graph_project(s, z_target, vis, gamma: float,
                            max_face_iters: int, a_init=None):
    """Project ``(s, ā)`` onto ``graph(sparsemax) = {(s, a) : a = Π_Δ(s)}``.

    Solves  min ‖s̄ − s‖² + γ²‖ā_proj − ā‖²  s.t.  ā_proj = sparsemax(s̄),
    row-wise, by the fixed-support (active-face) closed form plus a bounded
    support refinement.

    Fixed-support closed form
    -------------------------
    On the face with support S (|S| = m), sparsemax is affine:
        a_S = L s_S + c,  L = I_m − (1/m)𝟙𝟙ᵀ,  c = 𝟙/m,   a_{S^c} = 0,
    with τ(s) = (Σ_S s_k − 1)/m. L is the centering projector on the support:
    symmetric idempotent, L𝟙 = 0, Lc = 0. The normal equations of
        min ‖s̄ − s‖² + γ²‖L s̄ + c − ā‖²
    are (I + γ²LᵀL) s̄ = s + γ²Lᵀ(ā − c); with LᵀL = L and Lc = 0 they
    collapse, and since (I + γ²L)⁻¹ = I − (γ²/(1+γ²)) L (idempotence again):

        s̄_S = s_S + κ · L(ā_S − s_S),      κ = γ²/(1+γ²).

    Consequences used below: Σ_S κL(·) = 0, so Σ_S s̄ = Σ_S s and τ(s̄) = τ(s)
    — the face solve never moves the threshold; the face output is
        ā_proj = L s̄ + c = a + κ (Lā − Ls) = a + κ[(ā − mean_S ā) − (a − 1/m)]
    on S (a_S = Ls_S + 1/m ⇒ Ls_S = a_S − 𝟙/m). If ā = a then s̄ = s
    (idempotent).

    The face is a POLYHEDRON, not the full affine span: validity requires
    a ≥ 0 on S and s̄_k ≤ τ off S. The off-support inequality separates per
    coordinate (τ depends only on S-coordinates), so its exact projection is
    the elementwise clamp s̄_k = min(s_k, τ) — the (a_k = 0, s̄_k = τ) boundary
    of the face. This clamp is what keeps a *dropped* coordinate dropped:
    removing k from S lowers the threshold by (τ − s_k)/(m−1) < 0, so the raw
    score s_k would sit above the new τ and sparsemax(s̄) would re-activate it;
    the clamp pays the (s_k − τ)² it owes and restores consistency.
    Off-support target components ā_k, k ∉ S, cannot be matched by this face —
    they only exert pressure to change the support:

    Support-change handling (bounded branch check)
    ----------------------------------------------
    Full face enumeration is exponential; instead, after each fixed-support
    solve the KKT conditions of the resulting point are checked
    (s̄ must still induce S: ā_proj > 0 on S; s̄_k − τ ≤ 0 off S — the latter
    holds automatically since s̄ = s and τ is unchanged there):
      * drop coordinates driven negative (ā_proj_k < 0, k ∈ S);
      * add allowed off-support coordinates with positive target pressure —
        the two-branch comparison of the ReLU card, one coordinate at a time:
        on the enlarged face the solve would assign coordinate k the affine
        output (1−κ)(s_k − τ) + κ(ā_k − mean_S ā + 1/m); add k exactly when
        that predicted output is positive (evaluated with the current-face
        τ, mean, m — a one-coordinate perturbation estimate). Predicting the
        post-solve output directly (rather than comparing raw energies
        γ²ā_k² vs (τ−s_k)²) keeps the enlarged face self-consistent: a
        coordinate is only activated if the re-solve will actually give it
        positive weight.
    Then re-solve on the adjusted support; ``max_face_iters`` bounds the
    number of solves (default 2 = the "re-solve once, and stop" recipe).
    Rows are protected from emptying (a support of ≥1 always remains). The
    loop is a fixed-iteration tensor program — no data-dependent Python
    branching, hence no GPU→CPU sync in the training backward.

    T = 1 (or any single-entry allowed set): m = 1 ⇒ L = 0 ⇒ s̄ = s and
    ā_proj = 1 exactly — the graph projection degenerates to the
    identity-consistent solve, which is why the fused path needs no seqlen==1
    bail-out in sparsemax mode.

    Args: ``s`` finite scores (garbage allowed at ~vis), ``z_target`` = ā,
    ``vis`` bool allowed mask, ``gamma`` = output stiffness (the matmul cards'
    g convention; target_penalty = 1/γ² on the ā side), ``a_init`` optional
    precomputed sparsemax(s) (skips one sort).
    Returns ``(s_bar, tau)``: the projected scores (equal to s at ~vis
    entries) and the final face threshold τ (..., 1); the face output ā_proj
    is (s̄ − τ) on the final support, 0 off it — (s̄ − τ)₊ wherever the face
    is consistent. Not materialized: V2 needs only s̄.
    """
    kappa = (gamma * gamma) / (1.0 + gamma * gamma)
    safe_s = torch.where(vis, s, torch.zeros_like(s))
    zt = torch.where(vis, z_target, torch.zeros_like(z_target))
    if a_init is None:
        a_init, _ = sparsemax_forward(safe_s, vis)
    S = (a_init > 0) & vis

    n_solves = max(int(max_face_iters), 1)
    for it in range(n_solves):
        Sf = S.to(safe_s.dtype)
        m = Sf.sum(dim=-1, keepdim=True).clamp(min=1.0)
        sum_s = (safe_s * Sf).sum(dim=-1, keepdim=True)
        tau_S = (sum_s - 1.0) / m
        mean_t = (zt * Sf).sum(dim=-1, keepdim=True) / m
        # L(ā − s) restricted to S: center both on the support.
        delta = kappa * (((zt - mean_t) - (safe_s - tau_S - 1.0 / m)) * Sf)
        s_bar = safe_s + delta                      # off-support: s̄ = s
        if it == n_solves - 1:
            break
        # KKT check + bounded branch adjustment (no .any() early-out: keeps
        # the loop sync-free; extra iterations on converged rows are no-ops).
        drop = S & (safe_s - tau_S + delta < 0)     # face output L s̄ + c < 0
        pred = (1.0 - kappa) * (safe_s - tau_S) \
            + kappa * (zt - mean_t + 1.0 / m)
        add = vis & ~S & (zt > 0) & (pred > 0)
        S_new = (S & ~drop) | add
        # Never empty a row's support (a causal row always has >= 1 allowed).
        keep_old = S_new.sum(dim=-1, keepdim=True) == 0
        S = torch.where(keep_old, S, S_new)
    # Off-support threshold clamp (the polyhedral part of the face; see
    # docstring): any allowed off-support score above the final τ is pulled
    # down to it, otherwise sparsemax(s̄) would re-activate it.
    s_bar = torch.where(vis & ~S & (s_bar > tau_S),
                        tau_S.expand_as(s_bar), s_bar)
    return torch.where(vis, s_bar, s), tau_S


class SparsemaxProjection(torch.autograd.Function):
    """Exact projection node for ``a = sparsemax(s)`` (legacy-path card).

    Drop-in for SoftmaxProjection under config.attention_activation ==
    "sparsemax": forward is the masked sparsemax (respecting -inf entries from
    MaskedAddProjection), backward is the exact Euclidean projection of
    ``(s, ā)`` onto the graph — see sparsemax_graph_project for the full
    derivation. Saves the input (like SoftmaxProjection) plus τ (one float per
    row); the support is reconstructed as ``s − τ > 0``, never materialized in
    forward.

    Output stiffness follows the matmul cards' γ = g convention
    (config.projection_g, overridable per-card via config.sparsemax_gamma —
    the face-lock fix, see activation_gamma(); the qk_match_g_to_omega
    rescale concerns the QK solve's ω/g ratio and does not apply here where
    ω = 1).

    TD repeater convention: this node's own repeater is skipped (as in the
    fused node, see attention_fused.py), while the mask node below repeats on
    the finite pre-mask logits. In graph TD scope the floor is still
    forwarded so the trace is not severed.
    """

    @staticmethod
    def forward(ctx, x):
        vis = (x != float("-inf"))
        a, tau = sparsemax_forward(x, vis)
        ctx.save_for_backward(x, tau)
        return a

    @staticmethod
    def backward(ctx, z):
        x, tau = ctx.saved_tensors
        vis = (x != float("-inf"))
        # Recompute the pattern from τ alone: -inf − τ = -inf clamps to 0.
        a = torch.clamp(x - tau, min=0.0)
        gamma = activation_gamma()
        s_bar, _ = sparsemax_graph_project(
            torch.where(vis, x, torch.zeros_like(x)), z, vis, gamma,
            int(config.sparsemax_max_face_iters), a_init=a,
        )
        x_bar = torch.where(vis, s_bar, x)
        if ops.DIAG is not None:
            disp = torch.where(vis, x_bar - x, torch.zeros_like(a)).abs()
            ops.DIAG.record(
                "sparsemax",
                support_frac=float(((a > 0) & vis).float().sum() /
                                   vis.float().sum().clamp_min(1)),
                z_min=float(z.min()),
                max_logit_disp=float(disp.max()),
                mean_logit_disp=float(disp.mean()),
            )
        td_forward(ctx, x)  # repeater skipped (see class docstring); graph floor forwarded
        return x_bar, None


# ── fused node ────────────────────────────────────────────────────────────────

def _exact_alpha(A, B):
    """α of an exact-transport card solve (projection_alpha_auto's power-of-two
    rescale, as in matmul_proj): these cards weight t by 1/(α − t²) themselves."""
    if not config.projection_alpha_auto:
        return config.projection_alpha
    ratio = (A.square().sum(-1).mean() / B.square().sum(-2).mean().clamp_min(1e-12)).clamp(1e-3, 1e6)
    return config.projection_alpha * torch.exp2(torch.round(torch.log2(ratio)))


def _deflate_norm(d, x, eps, dim=-1):
    """Exact-transport amplitude: _deflate_cap's deflation, then the norm along
    `dim` SET (not capped) to eps·‖x‖ — the per-entry projections fix only the
    direction of each move. Zero-residual rows stay exactly zero. ‖x‖² is
    reduced once for both steps."""
    xx = x.square().sum(dim, keepdim=True)
    d = d - (d * x).sum(dim, keepdim=True) / xx.clamp_min(1e-12) * x
    if eps > 0.0:
        d = d * (eps * xx.sqrt() / d.norm(dim=dim, keepdim=True).clamp_min(1e-12))
    return d


def _group_fanin(x, mean, group):
    """GQAConsensusProjection.backward given the mean of the per-head targets:
    the mean, or delta_sum's x + Σ_g (t_g − x) (config.branch_mode)."""
    return x + group * (mean - x) if config.branch_mode == "delta_sum" else mean


class FusedCausalAttentionProjectionV2(torch.autograd.Function):
    """One-node projection backward for causal self-attention.

    Inputs: post-norm/rotary q (B,H,T,hd), k and v in KV-head shape
    (B,KV,T,hd); output y (B,H,T,hd). GQA is folded into rows (head
    h = kv·G + g, so (B,H,T,·) -> (B,KV,G·T,·) is a view): q@kᵀ and P@v need
    no K/V copies, and a B-side solve over the G·T rows is exactly the group
    mean of the per-head solves (GQAConsensusProjection). Per-head B-side
    caps use the (B,KV,G,T,·) view.

    Saves q, k, v, y and one row statistic: lse (softmax) or τ (sparsemax;
    the support is reconstructed from the bit-identically recomputed scores,
    no exp/log in this mode). Structurally correct at T = 1.

    Per-card target deltas are recorded through delta_probe.RECORDER when it
    is active.
    """

    @staticmethod
    def forward(ctx, q, k, v, omega, g_qk, tag=None):
        B, H, T, hd = q.shape
        G = H // k.shape[1]
        neg_inf, vis = _causal_masks(T, q.device, q.dtype, G)
        ctx.sparse = config.attention_activation == "sparsemax"
        # Same op order as the legacy chain -> bit-identical forward.
        qk = (q.reshape(B, -1, G * T, hd) @ k.transpose(-2, -1)).div_(omega)
        if ctx.sparse:
            P, aux = sparsemax_forward(qk, vis)
        else:
            P = F.softmax(qk.add_(neg_inf), dim=-1)
            aux = torch.logsumexp(qk, dim=-1, keepdim=True)
        y = (P @ v).reshape(B, H, T, hd)
        ctx.save_for_backward(q, k, v, aux, y)
        ctx.omega, ctx.g_qk, ctx.tag = omega, g_qk, tag
        return y

    @staticmethod
    def backward(ctx, z_y):
        q, k, v, aux, y = ctx.saved_tensors
        B, H, T, hd = q.shape
        KV = k.shape[1]
        G = H // KV
        omega, g_qk, sparse = ctx.omega, ctx.g_qk, ctx.sparse
        tag = ctx.tag if ctx.tag is not None else "?"
        exact = config.attn_exact_transport
        eps_lin = config.td_eps_lin
        eps_t = config.attn_transport_eps_scale * eps_lin
        cubic = config.proj_out_cubic
        mm = dict(alpha=config.projection_alpha, alpha_auto=config.projection_alpha_auto,
                  num_steps=config.attn_pairwise_num_steps, compute_z_proj=False,
                  b_self_deflate=config.matmul_b_self_deflate, out_cubic=cubic)
        td_active = (config.td_lambda != 0.0 and config.td_mode == "norm"
                     and TD_TRACE["g"] is not None)
        record = RECORDER.active()
        _, vis = _causal_masks(T, q.device, q.dtype, G)

        def heads(x):  # (B,KV,G·T,·) -> per-head (B,KV,G,T,·) view
            return x.unflatten(2, (G, T))

        q, y, z_y = (t.reshape(B, KV, G * T, hd) for t in (q, y, z_y))
        kT = k.transpose(-2, -1)
        if record:
            RECORDER.record(f"attn{tag}.y", delta=(z_y - y).norm(), x_norm=y.norm())

        # Recompute the logits (raw = A@B is the QK solve's exact p_init) and
        # the pattern from the saved row statistic, 0 at masked entries.
        raw = q @ kT
        qk = raw / omega
        P = (qk - aux).clamp_(min=0.0) if sparse else (qk - aux).exp_()
        P.masked_fill_(~vis, 0.0)

        # ── PV solve ──────────────────────────────────────────────────────────
        if exact in ("pv", "both"):
            # Per-entry sum transport (config.attn_exact_transport): the
            # regular per-entry bilinear projection, WITHOUT the consensus
            # average. Every product entry (i,j) keeps its own exact card —
            # the identical matmul_proj Newton solve — whose root t_ij gives
            # the per-copy displacements (appendix-5 closed form)
            #     δP_ij = [t²/(α−t²)]·P_i + [αt/(α−t²)]·v_j
            #     δv_ij = [t²/(α−t²)]·v_j + [ t/(α−t²)]·P_i .
            # The consensus card recombines by AVERAGING the full proposals
            # (1/T dilution — the min-norm scale that drowns in fp32 when
            # returned as A + δ); here the copies are recombined by SUMMING
            # their transported parts,
            #     dP_i = α Σ_j [t/(α−t²)]·v_j ,   dv_j = Σ_i [t/(α−t²)]·P_i ,
            # dropping the self parts, whose coefficients t²/(α−t²) ≥ 0 add
            # one-sidedly along the operand (the measured radial rider,
            # loss-null under the gauge P→cP, v→v/c). Any operand-parallel
            # component left in the transported sum is removed for the same
            # reason. The per-entry projections set the DIRECTION of each
            # row move only; its amplitude is NORMALIZED to the solver
            # fidelity window (_deflate_norm). The projection-scale sum
            # itself is ~1/q_eff-normalized per entry — orders of magnitude
            # below the window — and at that amplitude the transported
            # signal loses to the boundary weight cards' self-inflation
            # drift (measured: the needle bench stops learning).
            # num_steps=8: the transport consumes the per-entry root itself,
            # so it must be CONVERGED — the consensus card's 1-step estimate
            # (movement-clamped at |t| ≤ 0.5) saturates exactly on the
            # large-demand entries, compressing the signal the sum is
            # supposed to carry (measured: the needle bench stops learning
            # at num_steps=1).
            alpha = _exact_alpha(P, v)
            t = matmul_proj(P, v, z_y, alpha=alpha, g=config.projection_g, num_steps=8,
                            p_init=y if config.p_saved else None, compute_z_proj=False,
                            out_cubic=cubic)[3]
            t = t / (alpha - t.square())
            z_p = P + _deflate_norm(alpha * (t @ v.transpose(-2, -1)) * vis, P, eps_t)
            v_t = v + _deflate_norm(heads(P).transpose(-2, -1) @ heads(t), v[:, :, None], eps_t).mean(2)
            del t
        else:
            # Consensus card (legacy: pairwise divide-and-concur, omega=1).
            z_p, v_t = matmul_proj(P, v, z_y, g=config.projection_g,
                                   p_init=y if config.p_saved else None, **mm)[:2]
            if config.attn_value_adjoint:
                # Direct backprop-adjoint V target (ops.py value_solve block).
                v_t = v - _deflate_cap(heads(P).transpose(-2, -1) @ heads(y - z_y),
                                       v[:, :, None], eps_lin).mean(2)
            if config.attn_strip_self:
                # Remove the coherent operand-parallel rider (see
                # config.attn_strip_self): per P-row and per V-row.
                z_p = P + _deflate_cap(z_p - P, P, 0.0)
                v_t = v + _deflate_cap(v_t - v, v, 0.0)
        if td_active:
            if not sparse:  # masked entries carry no residual (T6; legacy MatMulProjection twin)
                z_p.masked_fill_(~vis, 0.0)
            z_p = td_bilinear_floor_update(P, z_p)
        if config.attn_prob_adjoint:
            # Exact PV adjoint for the pattern (ops.MatMulProjection twin).
            z_p = P - _deflate_cap(((y - z_y) @ v.transpose(-2, -1)) * (P != 0), P, eps_lin)
        if record:
            RECORDER.record(f"attn{tag}.pv", delta=(z_p - P).norm(), x_norm=P.norm())

        # ── activation card (+ mask target-replace on finite logits) ─────────
        if sparse:
            z_qk, _ = sparsemax_graph_project(qk, z_p, vis, activation_gamma(),
                                              int(config.sparsemax_max_face_iters), a_init=P)
        elif config.softmax_target_mode == "jacobian":
            z_qk = _softmax_mask_jacobian_impl(qk, z_p, P, vis)
        else:  # "rel_floor"
            z_qk = _softmax_mask_relfloor_impl(
                qk, z_p, P, vis,
                config.softmax_target_eps_rel, config.softmax_target_kappa,
                config.softmax_logit_clip, 1e-8,
            )
        supp = P > 0 if sparse and exact in ("qk", "both") else None
        del P, z_p
        # Mask-node repeater on the finite pre-mask logits (no-op at td_lambda 0).
        z_qk = td_repeat_target(qk, z_qk)
        if record:
            RECORDER.record(f"attn{tag}.act", delta=((z_qk - qk) * vis).norm(),
                            x_norm=(qk * vis).norm())
        s_z = (qk - z_qk) / omega if config.attn_qk_adjoint else None
        del qk
        z_qk.mul_(omega)  # the solve's target in A@B units

        # ── QK solve ──────────────────────────────────────────────────────────
        if exact in ("qk", "both"):
            # Per-entry sum transport (see the PV card above for the full
            # recipe): same matmul_proj per-entry solve, copies recombined by
            # summing the transported parts [αt/(α−t²)]·k_j per q-row and
            # [t/(α−t²)]·q_i per k-column. Masked entries carry
            # z_qk == qk, so their per-entry root is exactly 0 and they
            # contribute nothing. Deflating dq along each q-row and dkᵀ along
            # each k-column quotients the score gauge q→cq, k→k/c (loss-null
            # radial directions — the self parts and any transported
            # remainder along the operand).
            alpha = _exact_alpha(q, kT)
            t = matmul_proj(q, kT, z_qk, alpha=alpha, g=g_qk, omega=omega, num_steps=8,
                            p_init=raw if config.p_saved else None, compute_z_proj=False,
                            out_cubic=cubic)[3]
            del z_qk, raw
            if sparse:
                # Restrict to the support: sparsemax assigns off-support
                # scores weight exactly 0 (the face map is constant there),
                # so the off-support score targets encode support-CHANGE
                # pressure (the threshold clamp) — feasibility bookkeeping,
                # not demand on q/k; transporting it decays fidelity with T
                # (probe_exact_transport: cos 0.65→0.34 uncapped without
                # this mask).
                t.mul_(supp)
            t = t * vis / (alpha - t.square())
            q_t = q + _deflate_norm(alpha * (t @ k), q, eps_t)
            kT_t = kT + _deflate_norm(heads(q).transpose(-2, -1) @ heads(t), kT[:, :, None],
                                      eps_t, dim=-2).mean(2)
            del t
        else:
            # Consensus card (legacy: pairwise, omega=g_qk when qk_match_g_to_omega).
            q_t, kT_t = matmul_proj(q, kT, z_qk, g=g_qk, omega=omega,
                                    p_init=raw if config.p_saved else None, **mm)[:2]
            del z_qk, raw
            if config.attn_strip_self:
                # per q-row and per k-column (see the PV card)
                q_t = q + _deflate_cap(q_t - q, q, 0.0)
                kT_t = kT + _deflate_cap(kT_t - kT, kT, 0.0, dim=-2)
        if td_active:
            q_t = td_bilinear_floor_update(q, q_t)
        if config.attn_qk_adjoint:
            # Exact QK adjoints for q and kᵀ (ops.MatMulProjection twin).
            q_t = q - _deflate_cap(s_z @ k, q, eps_lin)
            kT_t = kT - _deflate_cap(heads(q).transpose(-2, -1) @ heads(s_z), kT[:, :, None],
                                     eps_lin, dim=-2).mean(2)
        if record:
            RECORDER.record(f"attn{tag}.qk", delta=(q_t - q).norm(), x_norm=q.norm())

        return (q_t.reshape(B, H, T, hd), _group_fanin(k, kT_t.transpose(-2, -1), G),
                _group_fanin(v, v_t, G), None, None, None)
