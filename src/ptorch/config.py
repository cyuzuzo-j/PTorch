import threading
from dataclasses import dataclass, field, fields, MISSING


@dataclass
class Config:
    """Projection framework configuration.

    Read settings as plain attributes:  config.use_projections
    Mutate with validation:             config.update("muon_weights", True)
    Temporary override:                 with config.projections(enabled=False): ...
    """

    # ---- cross-entropy constraint -------------------------------------------
    # lambda default 1.0 matches the long-standing effective behavior: the
    # CrossEntropy module historically never forwarded these knobs, so the
    # function defaults (5 steps, lambda=1.0) were what actually ran.
    cross_entropy_num_steps: int = 5
    cross_entropy_lambda: float = 1.0

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
    branch_mode: str = "mean"

    # ---- softmax target handling --------------------------------------------
    # "legacy": clamp(z, 1e-8) + renormalize + exact log-shift (can inject
    #           ~-18 log-targets when consensus targets go slightly negative).
    # "rel_floor": floor targets relative to current softmax output, then a
    #              damped, clipped log-space step.
    # "simplex_l2": Euclidean (sort-based) simplex projection first, then the
    #               rel_floor + damped log-space step.
    softmax_target_mode: str = "legacy"
    softmax_target_kappa: float = 1.0     # damping of the log-space step
    softmax_target_eps_rel: float = 0.05  # relative floor vs current weights
    softmax_logit_clip: float = 6.0       # max per-entry logit displacement

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
    #           dimension-safe; anneals to zero with the loss.
    td_mode: str = "vector"
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
    # window. Off by default.
    td_deflate: bool = False
    # Optional stability safeguard: when > 0, clamp the blended trace norm
    # (vector mode) or the floor (norm mode) to td_clip × ‖seed‖ per backward
    # sweep. Off (0.0) by default so the raw trace dynamics stay visible.
    td_clip: float = 0.0

    # ---- rmsnorm backward ----------------------------------------------------
    # "legacy": project target onto sqrt(n)-sphere, rescale x onto that ray.
    # "exact": joint L2 projection onto the RMSNorm graph manifold
    #          {(sigma*u, sqrt(n)*u) : sigma >= 0, ||u|| = 1}.
    rmsnorm_backward_mode: str = "legacy"
    rmsnorm_g: float = 1.0                # weight of the output-target term
    rmsnorm_newton_steps: int = 6
    
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
                if f.default is not MISSING:
                    object.__setattr__(self, f.name, f.default)

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
