import os
import threading
from dataclasses import dataclass, field, fields, MISSING
from typing import Optional, Tuple


def _env_bool(name: str, default: bool):
    """Field default seeded from env var `name` ("1" = on, anything else =
    off), so launch scripts that export the historical env flags keep working.
    Re-read on reset()."""
    return field(default_factory=lambda: os.environ.get(name, "1" if default else "0") == "1")


def _env_num(name: str, default, cast):
    return field(default_factory=lambda: cast(os.environ.get(name, str(default))))


def _env_str(name: str, default: str):
    return field(default_factory=lambda: os.environ.get(name, default))


def _env_ints(name: str, default: str):
    return field(default_factory=lambda: tuple(
        int(s) for s in os.environ.get(name, default).split(",") if s.strip()))


@dataclass
class Config:
    """Projection framework configuration.

    Read settings as plain attributes:  config.use_projections
    Mutate:                             config.update("td_lambda", 0.3)
    Restore defaults:                   config.reset()
    Record the full setting:            config.snapshot()

    update() rejects unknown keys; values are not checked.

    Fields declared with _env_* defaults are parsed from the named environment
    variable at `import ptorch` (and again on reset()), so a malformed value
    fails the import; env changes made later are ignored. After that they are
    ordinary keys: every consumer reads config per call, so config.update()
    takes effect immediately.

    Deliberately env-only (not config): PTORCH_COMPILE_DYNAMIC (consumed by
    @torch.compile decorators at import time).
    """

    # ---- cross-entropy constraint -------------------------------------------
    # Proximal CE: lambda = 10 with 1 inner step (was lambda = 1, 5 steps; the
    # shallow-MLP l_inf runs gain ~1 point on MNIST with 10/1).
    cross_entropy_num_steps: int = 1
    cross_entropy_lambda: float = 10.0
    cross_entropy_loss_value: bool = True   # False: CrossEntropy returns 0, not the loss (training is unchanged)

    # ---- global projection --------------------------------------------------
    use_projections: bool = True
    projection_alpha: float = 1.0
    projection_g: float = 1.0
    # Auto-balance the weight-vs-activation penalty per bilinear solve:
    # alpha_eff = projection_alpha * mean(qa)/mean(qb), where qa/qb are the
    # squared row/col norms of the activations/weights. With alpha=1 and wide
    # unnormalized layers (qa >> qb) the exact projection satisfies the
    # constraint almost entirely by moving the *weights* (cheap direction),
    # so the backward activation target dies at the first solve. Balancing
    # makes both directions comparably expensive, reviving signal flow.
    projection_alpha_auto: bool = False

    # ---- non-quadratic output-block generator (experimental, 2026-07-14) ------
    # Give the SCALAR output block h⁺ a non-quadratic Bregman generator while
    # keeping (h,θ) quadratic — the output is 1-D so its mirror map is O(1) and
    # the scalar root-find f(t)=0 is unchanged (efficiency preserved). The linear
    # map h̄⁺ = Z − t/γ² becomes h̄⁺ = Z − (1/γ²)(t + β·t³): curvature high near
    # t=0 (stiff → amplifies small/deep residuals) and softening for large |t|
    # (protects healthy top-layer residuals from over-inflation). This is the
    # "efficient yet genuinely non-Euclidean / potentially ℓ2-expansive" geometry
    # (see experiments/deep/FINDINGS_adaptive_alpha.md §non-quadratic output).
    # β = 0 (default) is bit-identical to the linear output map.
    proj_out_cubic: float = 0.0

    # ---- muon on activations ------------------------------------------------
    use_muon_activations: bool = False
    muon_activations_lr: float = 1.0
    # Norm floor in the polar-express normalization of activation residuals.
    # The default 1e-2 keeps sub-eps residuals small (direction-preserving),
    # but body residuals in deep nets sit at ~1e-7: they never reach the polar
    # factor and the rescue stalls. Lowering this (e.g. 1e-8) renormalizes any
    # real residual to unit scale so the Muon step has full magnitude.
    muon_activations_eps: float = 1e-2
    # How to size the activation-target update once direction is fixed.
    # "fixed":       polar(δA) × constant lr (legacy; fails at every lr tried —
    #                fixed magnitudes corrupt the head's genuine target).
    # "rel_row":     polar(δA) row-rescaled to lr × ||A_det row||. Per-token
    #                relative move is constant across layers, so the |B|-scaled
    #                shrinkage of δA in deep layers no longer kills the signal.
    # "rel_frob":    whole-tensor variant of rel_row (single norm ratio).
    # "raw_rel_row": skip polar, just per-row rescale of the natural δA — keeps
    #                matmul_proj's chosen direction, only fixes magnitude.
    muon_activations_mode: str = "fixed"
    # Skip the rescale for activations whose last_dim exceeds this (mode != "fixed"
    # only). Defaults to 4096 to exclude the flatten-head's 8192-dim input,
    # whose row norms are ~sqrt(8192) and would produce O(1) per-row
    # displacements — crushing the head's natural t~3e-2 target. Body layers
    # (last_dim ∈ {128, 512}) keep the rescue. 0 disables the gate.
    muon_activations_max_dim: int = 4096

    # ---- branch / fan-out target combination --------------------------------
    # "mean": consensus average of consumer targets (legacy).
    # "delta_sum": x_bar = x + sum_i (t_i - x) — backprop-residual analog,
    #              keeps gain-1 skip paths in deep residual nets.
    # STATUS 2026-07-12: both combines are metric-blind — inconsistent path
    # scalars at the fan-in destroy the merged direction regardless of rule
    # (probe P3 cos 0.044 for both); fixed at trainer level by MERGE_RHO
    # fan-in equalization (transport_theory.md T2, no-VJP record 3.2465).
    branch_mode: str = "mean"

    # ---- softmax target handling --------------------------------------------
    # "legacy": clamp(z, 1e-8) + renormalize + exact log-shift (can inject
    #           ~-18 log-targets when consensus targets go slightly negative).
    # "rel_floor": floor targets relative to current softmax output, then a
    #              damped, clipped log-space step.
    # "simplex_l2": Euclidean (sort-based) simplex projection first, then the
    #               rel_floor + damped log-space step.
    # "jacobian":  exact softmax vector-Jacobian product (backprop's adjoint).
    #              The three log-space modes above deliver a well-aligned target
    #              at the softmax OUTPUT but a logit target decorrelated from the
    #              true adjoint (cos ~0.08), which the QK bilinear solve then
    #              transports as garbage into q,k — annihilating the query/key
    #              learning pathway in attention. This mode restores direction
    #              fidelity to the logits (magnitude is Muon/TD-restored). See
    #              experiments/attention/FINDINGS_attn_transport.md.
    # STATUS 2026-07-12: "jacobian" CONFIRMED KEEPER (in every winning stack);
    # the log-space modes are FALSIFIED for attention transport (preimage 1/σ
    # at the softmax bottleneck, transport_theory.md §3 Corollary B).
    softmax_target_mode: str = "legacy"
    softmax_target_kappa: float = 1.0     # damping of the log-space step
    softmax_target_eps_rel: float = 0.05  # relative floor vs current weights
    softmax_logit_clip: float = 6.0       # max per-entry logit displacement

    # ---- attention value (PV) adjoint transport ------------------------------
    # For the attention y = P @ V matmul solve ONLY: replace the joint bilinear
    # solve's V-side target with the EXACT backprop adjoint  δV = Pᵀ @ s_y
    # (s_y = output-side residual at y). The joint solve's B-side target dilutes
    # the transported adjoint as the key dimension N=T grows (the B-parallel
    # self-term accumulates over all M=T query rows while the transported term is
    # averaged), so the value-pathway target fidelity decays with sequence length
    # (cos 0.80@T=16 → 0.05@T=1024; see FINDINGS_attn_transport.md §4). The exact
    # adjoint is capped into the solver's fidelity window (td_eps_lin·‖V‖ per row)
    # and the V-parallel self-component is deflated per row (same math as
    # td_deflate). The P (softmax pattern) target is left on the joint solve.
    # Off by default (bit-identical: only the PV solve is touched, and only when
    # this flag is on). Applies to MatMulProjection solves flagged value_solve.
    attn_value_adjoint: bool = False

    # ---- attention demand-transit adjoints (depth-ladder 2026-07-11) ----------
    # The L1 depth-ladder isolated the projection/gradient gap to attention's
    # DOWNWARD demand transit (mlponly parity, attnonly +0.74; the weight cards
    # are fine — FREEZE_QK hurts). The joint solve's A-side target is a
    # 1/N-diluted transported term plus an A-parallel self-rider, and the
    # demand crosses two such hops (PV then QK) before reaching x, so its
    # direction rots (tok_emb demand cos 0.62→0.16). These flags replace the
    # remaining attention-internal transit targets with the EXACT backprop
    # adjoint, row-deflated and capped into the fidelity window — the same
    # treatment attn_value_adjoint applies to V. Weight cards keep their solve
    # targets. Both off by default (bit-identical).
    # attn_prob_adjoint: PV solve A-side (P pattern target) — δP = s_y @ Vᵀ.
    attn_prob_adjoint: bool = False
    # attn_qk_adjoint: QK solve, both sides — δq = s_z @ k, δkᵀ = qᵀ @ s_z.
    attn_qk_adjoint: bool = False

    # ---- Hopfield-retrieval attention backward (ATTN_HOPFIELD=1) -------------
    # Attention as modern-Hopfield memory retrieval; backward = memory editing
    # (KL-prox retrieval retarget + Hebbian storage targets) replacing the
    # QK/mask/softmax/PV node composition entirely. Forward identical to the
    # legacy path. See ptorch/core/attention_hopfield.py. Off = bit-identical.
    attn_hopfield: bool = False
    # KL trust region for the retrieval retarget (per-row mean when tau scope
    # is "global"). Prototype: 1e-6–1e-4 improves retrieval in 82–99% of rows;
    # ≥1e-2 overshoots the linearization.
    attn_hopfield_eps_kl: float = 1e-4
    # "global" = one tau per tensor (preserves backprop's row weighting; exact
    # VJP limit, prototype cos 0.97). "row" = equal KL budget per row.
    attn_hopfield_tau: str = "global"
    # Inner KL-prox iterations: each step consumes eps_kl/iters of trust
    # region and recomputes delta from the updated retrieval (keeps every
    # step in the softmax linear regime while consuming a larger total
    # budget). 1 = legacy one-shot, bit-identical.
    attn_hopfield_iters: int = 1
    # qk_amp (2026-07-13): restore the tau factor on the q/k messages. The KL
    # trust region shrinks dq/dk ~ (1/tau)*softmax_jac(m) while dv is un-scaled,
    # inverting the gradient's q/k-vs-v amplitude ratio (probe: joint q/k/v
    # alignment 0.24 -> 0.74 with tau restored). Positive per-message scalar,
    # own-op/PJAX-legal. Default off (bit-identical legacy).
    attn_hopfield_qk_amp: bool = False
    # entmax (2026-07-13): replace the softmax attention forward with alpha-entmax
    # (Peters/Martins 2019) and the KL-prox retrieval backward with the Tsallis-
    # mirror prox. entmax is SPARSE (zeros the ~1024-key ill-conditioned tail that
    # gives softmax J=diag(p)-pp^T condition ~1e3-1e5); its Jacobian lives only on
    # the support -> ~350-890x better conditioned on the real patterns, which is
    # exactly what the projection (1/sigma) transport needs. alpha=1 recovers
    # softmax (runs the plain Hopfield softmax node), alpha=2 = sparsemax.
    # Default off (alpha ignored) = bit-identical.
    # REQUIRES attn_hopfield=True (entmax lives inside the Hopfield node). The
    # entmax backward is one-shot and un-amplified: it ignores
    # attn_hopfield_iters and attn_hopfield_qk_amp.
    attn_entmax: bool = False
    attn_entmax_alpha: float = 1.5
    # ReLU attention (2026-07-13): a=relu(qk^T/scale)/n (n=causal row length).
    # Diagonal 0/1-gate Jacobian -> condition EXACTLY 1, preimage==adjoint, so
    # the projection transport is EXACT through attention (probed cos 1.0 vs the
    # true VJP on real patterns). Own-op, PJAX-legal. Default off.
    # Dispatch priority (MultiheadAttention, both modes): attn_relu >
    # attn_hopfield > fused node (fused_attn) > legacy node composition; relu
    # and hopfield replace the whole core, so attention_activation is ignored.
    attn_relu: bool = False

    # ---- attention module knobs (formerly env-only) ----------------------------
    # Root-find steps of the consensus pairwise solves inside attention (QK and
    # PV MatMulProjection / matmul_proj). Read by every attention path — legacy
    # composition and the fused consensus cards — so fused and unfused agree.
    # (The V2 exact-transport cards use their own fixed 8 steps: they consume
    # the per-entry root directly.)
    attn_pairwise_num_steps: int = 1
    # QK_GAMMA (2026-07-11, standing-pattern-demand finding): multiply the QK
    # pairwise solve's gain g. target_penalty = ω²/g² in matmul_proj, so γ>1
    # makes the solve allocate MORE per-step movement to q,k for the same score
    # demand. Analog of sparsemax_gamma. 1.0 = bit-identical.
    qk_gamma: float = _env_num("QK_GAMMA", 1.0, float)
    # ATTN_TEMP (2026-07-13): forward softmax temperature. scale := sqrt(hd)*temp,
    # flatter rows -> better-conditioned softmax Jacobian for the projection
    # transport, at a forward-capacity cost. 1.0 = bit-identical.
    attn_temp: float = _env_num("ATTN_TEMP", 1.0, float)
    # ATTN_XQK_STOP: diagnostic — cut the q/k branch demand into x (identity
    # forward, delta=0 backward). Projection mode only.
    attn_xqk_stop: bool = _env_bool("ATTN_XQK_STOP", False)
    # ATTN_TRANSPORT_EPS_SCALE: transport-only amplitude knob of the V2
    # exact-transport cards — cap = SCALE * td_eps_lin * ‖row‖ without touching
    # td_eps_lin (shared by other cards). 1.0 = bit-identical.
    attn_transport_eps_scale: float = _env_num("ATTN_TRANSPORT_EPS_SCALE", 1.0, float)

    # ---- fused attention node (PTORCH_FUSED_ATTN) -----------------------------
    # One autograd node for GQA-expand -> QK solve -> mask -> softmax -> PV
    # solve (FusedCausalAttentionProjectionV2 in core/attention_native.py).
    # Applies only inside its config envelope (fused_attention_enabled);
    # outside it the module falls back to the legacy composition.
    # attn_exact_transport and attn_strip_self exist ONLY in the fused node, so
    # they need fused_attn=True and the envelope; otherwise they are inert.
    fused_attn: bool = _env_bool("PTORCH_FUSED_ATTN", False)
    # PTORCH_P_SAVED: warm-start the pairwise root-find from the saved forward
    # product (y / qk·ω) instead of recomputing it. Default on.
    p_saved: bool = _env_bool("PTORCH_P_SAVED", True)
    # PTORCH_FUSED_SOFTMAX: compiled rel_floor softmax backward. Default on.
    fused_softmax: bool = _env_bool("PTORCH_FUSED_SOFTMAX", True)

    # ---- trunk demand-transit adjoint (two-channel split, 2026-07-11) --------
    # With the Hopfield interior in place the attention cards get near-exact
    # targets GIVEN a true incoming delta, but the delta itself still crosses
    # every Linear card via the consensus A-side (1/N dilution + rider) and
    # rots hop by hop (direction probe: q/k cos 0.76 attnonly -> 0.29 full L1;
    # tok_emb demand cos 0.07 at 8221x inflation). This flag splits the two
    # channels the joint solve conflates: the WEIGHT target B_proj keeps the
    # consensus solve; the INPUT-demand return A_ret is replaced with the exact
    # adjoint s_z @ B^T, A-parallel deflated and capped into the fidelity
    # window (same recipe as attn_value_adjoint). Non-pairwise solves only;
    # off = bit-identical.
    matmul_transit_adjoint: bool = False
    # transit_vjp: the full two-channel stack (trunk-transit probe 2026-07-11).
    # The transit channel carries the exact, UNCAPPED VJP of the displacement
    # (conservative amplitudes — per-hop caps/trust regions starve whole paths:
    # measured 3e4 skip-vs-attention inversion at the residual fan-in); every
    # weight card keeps forming its own windowed projection target locally.
    # Covers matmul A-side (uncapped adjoint), Hadamard b-side (delta*a) and
    # the embedding scatter-SUM. (The Hopfield node's exact-VJP return branch
    # was removed 2026-07-12 after the channel tested neutral.) Combine with
    # RMSNORM_MODE=adjoint; branch delta_sum is already the summing combine.
    # STATUS 2026-07-12: CONFIRMED KEEPER paired with transit_vjp_card_cap —
    # the VJP record stack (3.2033 = preconditioned backprop, 5x walltime,
    # STATUS_LEDGER.md). Its value is cross-path CONSISTENCY, not per-channel
    # exactness (see transit_vjp_* attribution below).
    transit_vjp: bool = False
    # transit_vjp_card_cap: with transit_vjp, shrink the Z displacement each
    # non-pairwise card SOLVES against into the fidelity window with ONE global
    # per-tensor scalar (min(1, td_eps_lin*|out|/|dZ|)). The inter-node transit
    # (A_ret) keeps the raw conserved VJP — only the card's consumption is
    # bounded. Raw tvjp rejected at L1 (4.061 vs bar 3.564, w_max drift 2.0):
    # unbounded card demands over-rotate the weights; per-row caps are what
    # broke the fan-in balance in the first place, hence global-scalar.
    # STATUS 2026-07-12: CONFIRMED KEEPER — load-bearing half of the transit_vjp
    # pair (tvjp without card cap regresses 4.06 vs 3.33).
    transit_vjp_card_cap: bool = False
    # Diagnostic split of transit_vjp into per-node channels (2026-07-12), to
    # attribute the no-VJP -> VJP gap. Each ORs with transit_vjp so the global
    # flag is unchanged; individually they enable the exact VJP for ONE node
    # class only. matmul = linear A-side s_z@B^T (ops.py); embed = tok_emb
    # scatter-SUM (ops.py). The attn channel (Hopfield ds) was removed with its
    # code path after testing neutral (outcome below). The Hadamard/Scale
    # channel is inert under sfreeze (SCALAR_LR=0), so it is not split out.
    # OUTCOME 2026-07-12 (on the 3.4409 no-VJP base, ctrl 3.4448): attn alone
    # 3.4388 NEUTRAL; embed alone 3.4383 NEUTRAL; matmul alone 4.8694
    # CATASTROPHIC (uncapped w_max blowup); all three 4.577. Single-channel
    # covector exactness is NOT the active ingredient — cross-path consistency
    # at the merges is (STATUS_LEDGER.md; superseded as a lever by MERGE_RHO).
    transit_vjp_matmul: bool = False
    transit_vjp_embed: bool = False

    # ---- bilinear solve B-side self-term deflation ---------------------------
    # The matmul_proj reconstruction scales the B-side self-term by
    # alpha/(alpha - t^2) >= 1 — a pure inflation (t^2 >= 0, never cancels)
    # telling every weight column / key row to GROW along itself whenever any
    # paired t is nonzero. The transported (A^T t) term dilutes as 1/M with the
    # row count while the inflation does not, so at GPT scale (M = tokens or
    # T) the B target is dominated by self-growth: measured cos(pseudo-grad, W)
    # is persistently ~-0.24 for c_k (B-side of the T-row QK solve) and ~-0.1
    # for lm_head/attn.proj, driving the linear w_max runaway and Scale sign
    # flips in full-size runs. td_deflate strips the analogous artifact from
    # ACTIVATION (A-side) targets and attn_value_adjoint from the PV V-target;
    # this flag completes the set for all remaining B-side targets: the self
    # coefficient is replaced by exactly 1, so the B target carries only the
    # transported adjoint term B + (1/M) A^T (t/(alpha-t^2)) — a per-entry
    # reweighted backprop adjoint. Off by default (bit-identical).
    matmul_b_self_deflate: bool = False

    # ---- attention core (score→weights activation and its cards) -------------
    # Activation of the attention core: "softmax" (default) or "sparsemax"
    # (exact graph-projection card, both legacy and fused paths; see
    # core/attention_native.py).
    attention_activation: str = "softmax"

    # Bounded face refinement of the sparsemax graph projection: number of
    # fixed-support solves (>= 1). 2 = solve, refine support once, re-solve.
    sparsemax_max_face_iters: int = 2

    # Output stiffness (γ) of the sparsemax activation card ONLY. None =
    # use projection_g (bit-identical legacy behavior). Face-lock/dead-zone
    # fix (2026-07-09 audit): at γ=1 the exact graph projection prices a face
    # change (move an off-support score up to the threshold, cost gap²)
    # against ignoring the pattern demand (cost γ²·m², m = the PV-diluted
    # demand) and correctly refuses — freezing retrieval exactly where the
    # true sparsemax gradient is exactly 0 (dead zone). Raising γ on this
    # card alone makes demanded face changes affordable: γ=4 flips a gap-0.2
    # off-support lock in ~200 target iterations and speeds within-support
    # convergence ~2x, without stiffening any other solve (unlike raising the
    # global projection_g). Regression:
    # attention_does_attention_test.py::test_sparsemax_gamma_unlocks_face_lock.
    sparsemax_gamma: Optional[float] = None

    # Exact first-order transport at the attention-internal bilinear cards
    # (fused V2 only): "off" (default, bit-identical), "qk", "pv", "both".
    # Replaces the divide-and-concur consensus reconstruction of matmul_proj
    # — whose concur (averaging) step is only tight when the per-row proposals
    # are redundant (true for weights, false for attention: each of the T rows
    # is a DIFFERENT token's constraint; fidelity cos 0.80@T=16 → 0.05@T=1024,
    # FINDINGS_attn_transport §4) — with the local adjoints of each card's own
    # output residual:
    #     PV card:  s_y = y − z_y,          δP = s_y Vᵀ,   δV = Pᵀ s_y
    #     QK card:  s = (S − z_s)·ω,        δq = s K,      δKᵀ = qᵀ s
    # No 1/M dilution, no B-parallel self-inflation. Each adjoint is deflated
    # along its own operand (score-gauge quotient: q→cq, k→k/c leaves QKᵀ
    # invariant) and capped into the solver fidelity window (td_eps_lin·‖row‖).
    # Direction only: TD floor amplitude restoration is unchanged. Boundary
    # Linear solves (c_q/c_k/c_v/proj, weights) keep the consensus card, whose
    # averaging is legitimate denoising there.
    attn_exact_transport: str = "off"

    # Strip the coherent operand-parallel (self) term from the consensus
    # card's emitted displacement at the QK and PV cards, for all four
    # operands (per q-row, k-column, P-row, V-row). Measured
    # (bench_block_needle, fp64): a persistent ~20-25% one-sided
    # operand-parallel rider at every T — the radial DC that integrates into
    # the w_max drift, loss-null in direction; stripping zeroes it exactly
    # while moving the per-step direction by < 0.05 cos. NOTE: does NOT fix
    # the fp32 fidelity collapse with T (a precision effect — the consensus
    # min-norm scale emits ~1e-6-relative displacements that drown in fp32
    # when returned as A + δ; adjoint-form attn_exact_transport does fix it).
    # Generalizes attn_value_adjoint's deflate from the V operand to all
    # four. Consensus branches only. Default False = bit-identical.
    attn_strip_self: bool = False

    # ---- TD(λ) eligibility trace over depth (experimental) -------------------
    # Blends each hidden-state target with a trace seeded at the loss-node
    # residual: g ← td_lambda·g + (1−td_lambda)·r_local, and the blended target
    # h − g is passed upstream instead of the pure local projection target.
    # 0.0 = pure local targets (current behavior; the trace code path is
    # skipped entirely). Only applies where the trace width matches the layer
    # width — mismatched widths fall back to the pure local target (no
    # dimension bridge). Distinct from the proximal cross_entropy_lambda.
    # Chain graphs only: the trace relies on autograd's sequential
    # reverse-topological backward order.
    # STATUS 2026-07-12: TD floors are positive per-tensor scalars — legal
    # direction-free amplitude repair, provably unable to corrupt directions
    # (scalar-freedom theorem, transport_theory.md §4) and equally unable to
    # fix relative path weights. Retired under TRANSIT_VJP (0.0 = tie, −15%
    # step time); TD_LAMBDA=0.3 (norm+deflate) remains load-bearing in the
    # 3.2465 no-VJP record stack.
    td_lambda: float = 0.0
    # Trace variant.
    # "vector": Variant A — blend the residual *vector* down the sweep and pass
    #           the blended target upstream. Restores magnitude but the copied
    #           vector is frame-misaligned (cos ~ 1/sqrt(d) with the useful
    #           direction); kept as the measured negative control.
    # "norm":   Variant B — carry only a *scalar* floor down the sweep
    #           (floor ← λ·floor + (1−λ)·‖r_local‖, seeded at ‖seed‖) and
    #           rescale the local, correctly-transported direction up to it:
    #           write max(1, floor/‖r_local‖)·r_local. Frame-invariant, hence
    #           dimension-safe; anneals to zero with the loss. Default; this
    #           is the paper's Floor(λ).
    td_mode: str = "norm"
    # Norm mode only: cap the restored amplitude at td_eps_lin × ‖A‖ of the
    # layer being targeted. The bilinear solve transports direction through Bᵀ
    # faithfully (cos ≈ 0.95) only for residuals small relative to the
    # activations — measured window ~1e-4..1e-1 relative; above it the solve
    # saturates and the activation move degenerates to an A-parallel rescale.
    # The cap keeps every hop inside that window ("direction pipe"): Muon
    # restores the update magnitude from direction alone, so the chain only
    # needs to carry direction above the numerical noise floor. 0 disables.
    td_eps_lin: float = 1e-2
    # Norm mode only: deflate the per-row component of the local residual that
    # is parallel to the activation itself before rescaling. That component is
    # the bilinear solve's self-rescaling artifact (the t²/(α−t²)·A term), not
    # transported signal — ~20-25% of residual energy even inside the fidelity
    # window. On by default.
    td_deflate: bool = True
    # Norm mode + global scope only: carry a PER-ROW floor vector down the trunk
    # instead of a single whole-tensor scalar. Each token row (the (B*T, d) row
    # identity that persists through norms/activations/residual adds) gets its
    # own floor: floor_row ← λ·floor_row + (1−λ)·‖r_row‖, capped per row at
    # td_eps_lin·‖A_row‖, and rescaled per row. Seeded per row at the loss
    # (‖seed_row‖). Fixes the whole-tensor granularity failure at GPT scale,
    # where one Frobenius floor / rescale shared by thousands of token rows of
    # wildly different difficulty leaves a large fraction of rows outside their
    # own fidelity window (measured: 48–85% of rows below 1e-4 rel-amplitude at
    # depth with the scalar floor; see FINDINGS_td_row.md). Attention-internal
    # matmuls (QK, AV) mix rows, so the per-row floor is COLLAPSED to its mean
    # there (documented semantics); the token-row floor is transported unchanged
    # around the attention block and resumes at the c_q/c_k/c_v input solves.
    # Off by default; λ=0 (or off) is bit-identical to the scalar path.
    td_row: bool = False
    # Optional stability safeguard: when > 0, clamp the blended trace norm
    # (vector mode) or the floor (norm mode) to td_clip × ‖seed‖ per backward
    # sweep. Off (0.0) by default so the raw trace dynamics stay visible.
    td_clip: float = 0.0
    # Trace scope (norm mode only).
    # "global": legacy single module-level trace updated in autograd-engine
    #           order. Correct only for sequential chain graphs. Bit-identical
    #           to the historical behavior.
    # "graph":  per-edge floor registry keyed by the autograd grad_fn node, so
    #           each node sees the floor of its own downstream path. Required for
    #           dense DAGs (attention branches, U-Net skips, residual fan-ins).
    #           Ignored when td_mode != "norm".
    td_scope: str = "global"
    # Graph scope only: how fan-out floors meet at a shared producer.
    # "max": skip-dominant (default). "mean": average the incoming floors.
    td_join: str = "max"

    # ---- rmsnorm backward ----------------------------------------------------
    # "legacy": project target onto sqrt(n)-sphere, rescale x onto that ray.
    # "exact": joint L2 projection onto the RMSNorm graph manifold
    #          {(sigma*u, sqrt(n)*u) : sigma >= 0, ||u|| = 1}.
    # "detached": per-token linear transport x_bar = z_target * rms(x) (scale
    #          treated as a constant). Exactly invertible: cannot vanish or
    #          renormalize a target; drops the Jacobian's x-parallel term.
    # ("tangent"/"adjoint": see RMSNormProjection, ops.py.)
    # STATUS 2026-07-12: "adjoint" works only WITH the transit-VJP chain (VJP
    # record 3.2033) or merge-equalized fan-ins (MERGE_RHO); standalone
    # off-VJP FALSIFIED 3.9768 (+0.35). "tangent" = off-VJP workhorse
    # (3.2465 record stack). "legacy" superseded: sign flips + full-magnitude
    # renormalization of vanished targets.
    rmsnorm_backward_mode: str = "legacy"
    rmsnorm_g: float = 1.0                # weight of the output-target term
    rmsnorm_newton_steps: int = 6

    # ---- Hadamard (Scale) projection metric ----------------------------------
    # "l2" = fixed-a L2 projection b + a(z - ab)/(1 + a^2) (bounded, no floor
    # needed). "linf" = Chebyshev projection
    # onto the hyperbola (a+da)(b+db)=z with max(|da|,|db|) minimized: bounded
    # b-demand (~sqrt(|r|) as a->0) with no floor, direction-changing. a-side
    # keeps the batch-reduced least-squares card target either way.
    hadamard_norm: str = "l2"

    # ---- embedding consensus reweighting -------------------------------------
    # Row target for EmbeddingProjection: t_v = w_v + count_v^beta * (mean_v - w_v).
    # beta=0 (default) = scatter-mean (pure consensus, bit-identical legacy);
    # beta=1 = scatter-sum (conservative demand: every occurrence contributes
    # its full displacement). Per-row positive scalars — direction-preserving
    # per row; tests whether ProjectionAdam's per-coordinate normalization
    # absorbs token-frequency reweighting (predicted mostly yes).
    embed_count_beta: float = 0.0


    # ---- TD implementation switches (formerly env-only) ------------------------
    # PTORCH_TD_SIBLING_SHARE: ratio-preserving TD rescale across sibling cards
    # (q/k/v Linears of one attention module), using a one-step-stale joint
    # norm.
    td_sibling_share: bool = _env_bool("PTORCH_TD_SIBLING_SHARE", False)

    # ---- MatMulProjection backward switches (formerly env-only) ----------------
    # PTORCH_MM_PROX_MU > 0: replace the A-side transport of 2D Linear cards with
    # the LM/damped-prox message ΔA = μ' rBᵀ(I + μ'BBᵀ)⁻¹, μ' = μ·K/tr(BBᵀ).
    # μ→0 = scaled adjoint, μ→∞ = min-norm preimage. 0 = off (bit-identical).
    mm_prox_mu: float = _env_num("PTORCH_MM_PROX_MU", 0.0, float)
    # Restrict the μ-prox override to cards run inside mm_prox_site(<scope>)
    # ("attn" = c_q/c_k/c_v/attn.proj, in both attention modules) or "all".
    mm_prox_scope: str = _env_str("PTORCH_MM_PROX_SCOPE", "all")
    # Keep the heuristic pipeline's per-tensor message amplitude and swap only
    # the direction to the Gram-damped adjoint (needs mm_prox_mu > 0).
    mm_prox_dirswap: bool = _env_bool("PTORCH_MM_PROX_DIRSWAP", False)
    # PTORCH_LINF_CHUNK_ELEMS: row-chunk budget (elements) of the l∞ matmul
    # solve. Memory knob only; numerics are chunk-invariant.
    linf_chunk_elems: int = _env_num("PTORCH_LINF_CHUNK_ELEMS", 2 ** 26, int)
    # Safeguarded Newton steps of the l∞ radius solve, used by Linear(norm='linf')
    # when num_iters is not given. 1 = the old first-order (linearized) radius
    # (default; 5 tightens the radius at ~1.3x the l∞ solve time).
    linf_num_iters: int = 1
    # Start each l∞ radius solve from the layer's previous radius (same shape)
    # instead of eps = 0, so with few Newton steps the radius keeps converging
    # across batches. This was the behavior before 4de0227e9 (Linear's
    # use_cache=True default).
    linf_warm_start: bool = False
    # The l∞ Newton step as it was before 4de0227e9/6637ae9cf: the
    # term_A >= term_B branch test (left derivative at the eps = 0 tie) and an
    # unbracketed clamp(eps - step, 0).
    linf_legacy_newton: bool = False

    # ---- trunk demand governor / de-dilution (core/governor.py, dedilute.py) --
    # DEMAND_GOVERNOR: per-row trust region cap = governor_tau·‖x‖ on the
    # residual-stream demand entering the listed block boundaries.
    demand_governor: bool = _env_bool("DEMAND_GOVERNOR", False)
    governor_tau: float = _env_num("GOVERNOR_TAU", 0.05, float)
    governor_boundaries: Tuple[int, ...] = _env_ints("GOVERNOR_BOUNDARIES", "0,1")
    # DEDILUTE_LINEAR: radial-deflate + N-gain (× dedilute_gain_scale) on the
    # demand entering attn.proj (both causal attention modules), window-capped.
    dedilute_linear: bool = _env_bool("DEDILUTE_LINEAR", False)
    dedilute_gain_scale: float = _env_num("DEDILUTE_GAIN_SCALE", 1.0, float)

    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def update(self, key: str, value) -> None:
        if key not in _PUBLIC_KEYS:
            raise AttributeError(
                f"Unknown config key {key!r}. Valid keys: {sorted(_PUBLIC_KEYS)}"
            )
        with self._lock:
            object.__setattr__(self, key, value)

    def reset(self) -> None:
        with self._lock:
            for f in fields(self):
                if f.name.startswith("_"):
                    continue
                if f.default is not MISSING:
                    object.__setattr__(self, f.name, f.default)
                elif f.default_factory is not MISSING:
                    object.__setattr__(self, f.name, f.default_factory())

    def snapshot(self) -> dict:
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if not f.name.startswith("_")
        }


_PUBLIC_KEYS: frozenset = frozenset(
    f.name for f in fields(Config) if not f.name.startswith("_")
)

config = Config()
