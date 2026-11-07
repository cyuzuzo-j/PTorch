import inspect
import os
import torch
import torch.nn.functional as F
from ..config import config
import math
from itertools import repeat

# PTORCH_COMPILE_DYNAMIC: 1 (default, symbolic shapes) / 0 (static-shape
# specialization; faster kernels when shapes are constant, recompiles per
# distinct shape) / auto (torch default).
_TC_DYNAMIC = {"1": True, "0": False, "auto": None}[os.environ.get("PTORCH_COMPILE_DYNAMIC", "1")]
# PTORCH_COMPILE_MODE: torch.compile mode of the hot-path compiled impls; unset
# (default) = torch's default mode. "reduce-overhead" replays each compiled call
# as a CUDA graph (cudagraph trees): far less host launch overhead per call, but
# it IS CUDA graphs, so label results accordingly. Use with
# PTORCH_COMPILE_DYNAMIC=0, and without an outer whole-step CUDA graph.
# The trainer MUST call torch.compiler.cudagraph_mark_step_begin() once per
# training step (each forward+backward), before the first backward: these
# impls run no-grad inside backward, so cudagraph trees would otherwise start
# a new generation on EVERY compiled call and free the previous call's
# outputs while the next node still reads them. Outputs that outlive the
# step (parameter residuals that AccumulateGrad may steal as p.grad) are
# copied out by _tc_own; TD floor state stays on the default mode.
# Under reduce-overhead, torch.cuda.max_memory_allocated() is NOT an accurate
# peak: replays reuse the cudagraph pools without allocator accounting.
_TC_MODE = os.environ.get("PTORCH_COMPILE_MODE") or None


def _tc_own(t):
    """Under _TC_MODE, an eager copy of a cudagraph-pool output that outlives
    the step (the next step's replay overwrites the pool). Identity otherwise."""
    return t.clone() if _TC_MODE is not None and t is not None else t

# Diagnostics sink. Scripts (e.g. experiments/attention/diag_targets.py) may
# set `ops.DIAG` to an object exposing `record(key: str, **scalars)`; the
# uncompiled autograd backward wrappers then emit target statistics. Must stay
# None in production — every instrumentation site guards on `DIAG is not None`
# so the default path is unchanged. Never read this inside @torch.compile
# bodies.
DIAG = None

# TD(λ) eligibility trace over depth (config.td_lambda). Seeded by td_seed
# from the loss-node residual in the loss projection's backward (which autograd
# runs first), then updated at each bilinear solve (td_bilinear_floor_update)
# as the engine walks the chain deep→shallow; every other projection node is
# an amplitude repeater (td_repeat_target). Module-level state is sound only
# for sequential chain graphs executed by a single backward at a time. "g" is
# the vector trace (td_mode "vector"), or just True in norm mode (the "trace
# seeded" gate); "seed_norm" backs the optional td_clip safeguard; "floor" is
# the scalar trace of norm mode (Variant B). Floors are on-device tensors, so
# the TD path never syncs with the host.
TD_TRACE = {"g": None, "seed_norm": None, "floor": None,
            # td_row (config.td_row, norm+global only): per-row floor vector
            # (M, 1) carried down the token-row trunk. seed_row backs td_clip.
            "floor_row": None, "seed_row": None}

# ---------------------------------------------------------------------------
# Graph-scope TD floors (config.td_scope == "graph", norm mode only).
#
# The chain-only TD_TRACE above leaks floors across parallel paths on a dense
# DAG. In graph scope we instead keep a per-edge registry keyed by the autograd
# grad_fn node: for a custom autograd.Function the `ctx` handed to backward *is*
# the backward node (BackwardCFunction) and a saved input tensor's `.grad_fn` is
# the producer's node, so a consumer can address its producer with no extra
# tensor plumbing. Entries are (floor, count) with 0-dim floors; the dict is
# cleared each time a loss backward seeds a new sweep (also prevents retaining
# graph nodes across steps).
TD_FLOORS = {}

# Single-input view/movement backward nodes we walk *through* to reach the real
# producer (matched by type name prefix). Anything multi-input (cat/split/add)
# stops the walk and drops the floor (degrade to pure-local == lambda 0).
_TD_VIEW_NODES = (
    "ViewBackward", "ReshapeAlias", "TransposeBackward", "PermuteBackward",
    "UnsqueezeBackward", "SqueezeBackward", "SliceBackward", "ToCopyBackward",
    "CloneBackward", "ContiguousBackward", "ExpandBackward",
    # F.pad's backward: a biased Linear pads its input with a ones column, so
    # ConstantPadNd sits on every trunk edge between consecutive matmul solves.
    "ConstantPadNd",
)


def _td_resolve(node, depth=8):
    """Walk up single-input view/movement nodes to the producing node."""
    for _ in range(depth):
        if node is None:
            return None
        if not type(node).__name__.startswith(_TD_VIEW_NODES):
            return node
        nexts = getattr(node, "next_functions", ())
        if len(nexts) != 1 or nexts[0][0] is None:
            return None
        node = nexts[0][0]
    return node


def td_seed(loss_input, seed):
    """Reset the whole trace and seed it from the loss residual `seed`.

    Called by every loss projection's backward when td_lambda != 0. Graph
    scope parks ‖seed‖ at the loss input's producer; td_row seeds the per-row
    floor; otherwise the global scalar floor (norm mode) or the vector trace.
    """
    TD_FLOORS.clear()
    TD_TRACE.update(g=None, seed_norm=None, floor=None, floor_row=None, seed_row=None)
    norm = config.td_mode == "norm"
    if norm and config.td_scope == "graph":
        TD_TRACE["seed_norm"] = seed.norm()
        node = _td_resolve(getattr(loss_input, "grad_fn", None))
        if node is not None:
            TD_FLOORS[node] = (TD_TRACE["seed_norm"], 1)
    elif norm and config.td_row:
        TD_TRACE["floor_row"] = TD_TRACE["seed_row"] = \
            seed.reshape(-1, seed.shape[-1]).norm(dim=-1, keepdim=True)
    else:
        TD_TRACE.update(g=True if norm else seed, seed_norm=seed.norm())
        TD_TRACE["floor"] = TD_TRACE["seed_norm"]


def td_get(ctx):
    """Pop this node's incoming floor (0-dim tensor) or None if it has none."""
    if config.td_scope != "graph" or config.td_mode != "norm":
        return None
    floor, n = TD_FLOORS.pop(ctx, (None, 1))
    return floor if n == 1 else floor / n


def td_send(saved_input, floor):
    """Route `floor` (0-dim tensor) to the producer of `saved_input`.

    Fan-out floors meeting at one producer combine per td_join: "max", or the
    order-independent "mean" (sum and count, divided at td_get). Drops
    silently (degrade to lambda 0 locally) if the producer cannot be resolved.
    """
    if floor is None:
        return
    node = _td_resolve(getattr(saved_input, "grad_fn", None))
    if DIAG is not None:
        DIAG.record("td_send", hit=1 if node is not None else 0)
    if node is None:
        return
    prev, n = TD_FLOORS.get(node, (None, 0))
    if prev is not None:
        floor = prev + floor if config.td_join == "mean" else torch.maximum(prev, floor)
    TD_FLOORS[node] = (floor, n + 1 if config.td_join == "mean" else 1)


def td_forward(ctx, *inputs):
    """Graph-scope TD pass-through for any projection node that is not a
    bilinear floor updater: pop this node's floor and forward it unchanged to
    the producers of `inputs` (default: the node's saved tensors; saved
    outputs, whose grad_fn is the node itself, are skipped). Returns the floor
    (None outside graph scope or when none arrived). One line per backward:

        td_forward(ctx)               # or td_forward(ctx, x) for chosen inputs

    td_repeat_target / process_node_target already call it."""
    floor = td_get(ctx)
    for t in inputs or ctx.saved_tensors:
        if t.grad_fn is not ctx:
            td_send(t, floor)
    return floor


# Default mode even under _TC_MODE: its floors persist across steps in
# TD_TRACE / TD_FLOORS / sibling state, so they must not live in a cudagraph pool.
@torch.compile(dynamic=_TC_DYNAMIC)
def _td_rescale_impl(x, x_bar, floor, cap, joint, lam: float, eps_lin: float,
                     deflate: bool, rows: bool, update: bool):
    """The one TD floor kernel, shared by every scope, the repeaters and the
    bilinear floor updaters. r = x - x_bar (x-parallel part stripped per row
    when `deflate`); norms are per last-dim row when `rows` (td_row), else
    whole-tensor. `update` λ-blends ‖r‖ into the floor (min'd with the td_clip
    `cap`). r is then lifted (never shrunk) to amp = min(floor, eps_lin·‖x‖),
    or by the common sibling factor amp/`joint`, except when ‖r‖ is at the
    rounding level of ‖x‖ (amplifying that noise gives a full-floor garbage
    target). Returns (target, floor, ‖r‖, scale)."""
    shape = x.shape
    floor = torch.as_tensor(floor, dtype=x.dtype, device=x.device)  # floor state lives in x's dtype
    if rows:
        x, x_bar = x.reshape(-1, shape[-1]), x_bar.reshape(-1, shape[-1])
    r = x - x_bar
    if deflate:
        r = r - (r * x).sum(-1, keepdim=True) / x.square().sum(-1, keepdim=True).clamp_min(1e-12) * x
        x_bar = x - r
    r_norm = torch.linalg.vector_norm(r, dim=-1 if rows else None, keepdim=rows)
    x_norm = torch.linalg.vector_norm(x, dim=-1 if rows else None, keepdim=rows)
    if update:
        floor = lam * floor + (1.0 - lam) * r_norm
        if cap is not None:
            floor = torch.minimum(floor, cap)
    amp = torch.minimum(floor, eps_lin * x_norm) if eps_lin > 0.0 else floor
    noise = 16 * torch.finfo(x.dtype).eps * x_norm
    scale = torch.where(r_norm > noise, amp / ((r_norm if joint is None else joint) + 1e-12),
                        1.0).clamp_min(1.0)
    return torch.where(scale > 1.0, x - scale * r, x_bar).reshape(shape), floor, r_norm, scale


def _td_apply(x, x_bar, floor, update=False, rows=False, sibling=None):
    """Host-side wrapper of _td_rescale_impl: td_clip cap, sibling share, DIAG.
    Returns (target, floor). Bilinear updaters (`update`) deflate per
    config.td_deflate; repeaters never do.

    Sibling share (config.td_sibling_share; the q/k/v cards of one attention
    module, tagged by td_sibling_group): rescale all siblings by ONE common
    factor amp·sqrt(n)/joint, the joint residual norm of the previous step
    (positive scalars: staleness is safe and every sibling reads the same
    joint, so the q:k:v ratios are preserved exactly within a step)."""
    seed = TD_TRACE["seed_row" if rows else "seed_norm"]
    cap = config.td_clip * seed if update and config.td_clip > 0.0 and seed is not None else None
    joint = None
    if sibling is not None and sibling.get("joint") is not None:
        joint = sibling["joint"] / sibling["size"] ** 0.5
    out, floor, r_norm, scale = _td_rescale_impl(
        x.detach(), x_bar.detach(), floor, cap, joint, float(config.td_lambda), float(config.td_eps_lin),
        update and config.td_deflate, rows, update)
    if sibling is not None:
        cur = sibling.setdefault("cur", [])
        cur.append(r_norm)
        if len(cur) >= sibling["size"]:
            sibling["joint"] = torch.stack(cur).norm()
            sibling["cur"] = []
    if DIAG is not None and update and not rows:
        DIAG.record("td_trace", r_local_norm=float(r_norm),
                    g_norm=float(scale * r_norm), scale=float(scale))
    return out, floor


_POLAR_COEFFS = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375)
]

_POLAR_COEFFS = [
    (a / 1.01, b / 1.01**3, c / 1.01**5) for a, b, c in _POLAR_COEFFS[:-1]
] + [_POLAR_COEFFS[-1]]

@torch.compile()
def zeropower_via_polarexpress(G: torch.Tensor, steps: int = 5, eps: float = 1e-2) -> torch.Tensor:
    """
    Computes the polar factor using the optimal Polar Express polynomial method.
    Acts as a drop-in replacement for the static Newton-Schulz5 / Jordan method.
    """
    is_1d = G.ndim == 1
    if is_1d:
        G = G.view(1, -1)
        
    X = G.bfloat16()
    
    # Direction-preserving normalization: floor the *denominator* at eps rather
    # than adding eps. With the additive form a tiny-norm G (e.g. a body
    # activation residual ~1e-7) is scaled toward zero (G/(‖G‖+1e-2) ≈ 100·G),
    # collapsing the polar iteration. clamp_min keeps the unit direction for any
    # ‖G‖ > eps, so small-but-real targets survive. (For weight-Muon, ‖G‖≫eps so
    # this is behaviourally identical to the old additive form.)
    X = X / X.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)
    
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
        
    hs = _POLAR_COEFFS[:steps]
    if steps > len(_POLAR_COEFFS):
        hs += list(repeat(_POLAR_COEFFS[-1], steps - len(_POLAR_COEFFS)))
        
    for a, b, c in hs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
        
    out = X.mT if transposed else X
    
    if is_1d:
        out = out.view(-1)
        
    return out.to(G.dtype)

def process_activation_target(A_det, A_proj):
    if not config.use_muon_activations:
        return A_proj

    orig_shape = A_det.shape
    last_dim = orig_shape[-1]

    if last_dim == 0 or A_det.numel() == 0:
        return A_proj

    mode = config.muon_activations_mode
    max_dim = config.muon_activations_max_dim

    # Skip the head solve: its activation has |row(A)| ~ sqrt(8192*var) so a
    # per-row rescale to lr*|row(A)| produces O(1)-magnitude head displacements
    # that swamp the natural t~3e-2 head target the head needs to learn. The
    # body solves are where signal vanishes, so keep the rescue there.
    if max_dim > 0 and last_dim > max_dim and mode != "fixed":
        return A_proj

    A_2d = A_det.reshape(-1, last_dim)
    A_proj_2d = A_proj.reshape(-1, last_dim)

    # lr crosses into the compiled impl as a 0-dim tensor, not a Python float
    # read from config: dynamo guards on Python floats, so a per-step decay of
    # muon_activations_lr would otherwise recompile until the cache limit and
    # then fall back to eager. mode/eps only change between runs, so static
    # specialization on them is fine.
    lr_t = torch.tensor(
        config.muon_activations_lr, device=A_det.device, dtype=A_2d.dtype
    )
    A_proj_new = _process_activation_target_impl(
        A_2d, A_proj_2d, lr_t, mode, config.muon_activations_eps
    )
    return A_proj_new.reshape(orig_shape)


@torch.compile()
def _process_activation_target_impl(A_2d, A_proj_2d, lr, mode, eps):
    g = A_2d - A_proj_2d

    if mode == "rel_row":
        # Per-row relative magnitude: polar-orthogonalize the per-row
        # displacement direction, then scale each row to lr * ||A_det row||.
        # Decouples the activation-target signal magnitude from |B| (the
        # per-layer min-norm bound) — each token's update has the same
        # relative size across all layers, restoring usable body signal that
        # the |B|-scaled δA = t·Bᵀ/N would otherwise lose at init.
        g_muon = zeropower_via_polarexpress(g, eps=eps)
        a_row_norm = A_2d.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        g_row_norm = g_muon.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        update = lr * (a_row_norm / g_row_norm) * g_muon
    elif mode == "rel_frob":
        # Whole-tensor variant: preserve direction, rescale to lr * ||A||_F.
        g_muon = zeropower_via_polarexpress(g, eps=eps)
        update = lr * (A_2d.norm() / g_muon.norm().clamp_min(1e-12)) * g_muon
    elif mode == "raw_rel_row":
        # No polar, just per-row magnitude rescale of the natural δA direction.
        # Keeps the (presumed-correct) direction from matmul_proj; only fixes
        # magnitude. Cheaper than polar and avoids any direction-quality risk.
        g_row_norm = g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        a_row_norm = A_2d.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        update = lr * (a_row_norm / g_row_norm) * g
    else:
        # "fixed" legacy: polar-orthogonalized direction × constant lr.
        g_muon = zeropower_via_polarexpress(g, eps=eps)
        update = lr * g_muon

    return A_2d - update


# ── TD direction-pipe repeater (non-bilinear nodes) ──────────────────────────
def td_repeat_target(x_det, x_bar, ctx=None):
    """Rescale a non-bilinear node's input-target residual up to the current
    trace floor (capped into the solver's linear-transport window), without
    updating the floor — λ-blending happens only at the bilinear solves, so
    activation/pooling/norm nodes act as amplitude repeaters between them.
    Exact no-op at td_lambda = 0 or outside norm mode.

    In graph scope the floor is read from the per-edge registry keyed by this
    node's `ctx` and forwarded unchanged to the producers of the node's saved
    inputs (td_forward). td_row uses the per-row floor (collapsed to its mean
    when the node's rows are not the token rows); otherwise the global chain
    floor. `x_det` must be finite (see SoftmaxProjection)."""
    if config.td_lambda == 0.0 or config.td_mode != "norm":
        return x_bar
    rows = False
    if config.td_scope == "graph":
        floor = td_forward(ctx)
    elif config.td_row and TD_TRACE["floor_row"] is not None:
        floor, rows = TD_TRACE["floor_row"], True
        if x_det.numel() // x_det.shape[-1] != floor.shape[0]:
            floor = floor.mean()
    else:
        floor = TD_TRACE["floor"]
    return x_bar if floor is None else _td_apply(x_det, x_bar, floor, rows=rows)[0]


def process_node_target(x_det, x_bar, ctx=None):
    """process_activation_target followed by the TD repeater; used by every
    non-bilinear projection backward (the bilinear matmul solves keep the raw
    process_activation_target and run the full floor update instead). Pass the
    backward `ctx` so graph scope can route the floor along the real edge."""
    out = process_activation_target(x_det, x_bar)
    return td_repeat_target(x_det, out, ctx)



# config.mm_prox_mu > 0 (env PTORCH_MM_PROX_MU) replaces the A-side transport of the trainer's 2D
# Linear cards with the LM/damped-prox message (vanishing_targets_theory §3.4
# ladder, the "μ dial"): ΔA = μ' rBᵀ(I + μ'BBᵀ)⁻¹ with r = (Z_target − out)·ω
# and μ' = μ·K/tr(BBᵀ) (dimensionless dial; per-direction gain μ'σ/(1+μ'σ²)).
# μ→0 recovers the scaled adjoint (σ weighting), μ→∞ the min-norm preimage
# (1/σ weighting); interior μ is the theory-predicted optimum. Own-op local
# Gram only — PJAX-legal, no covector chain — and damping-bounded (max
# per-direction gain ≤ ½√μ'), so it cannot hit the amplitude-drift failure
# mode. B-side weight update, TD floors, and pairwise/attention solves are
# untouched.

# config.mm_prox_scope="attn" (env PTORCH_MM_PROX_SCOPE) restricts the μ-prox override to Linear cards
# executed inside an `mm_prox_site("attn")` region (the four attention cards
# c_q/c_k/c_v/attn.proj — the cond-1e6 anisotropic hops the 2026-07-12 L2
# transit probe blames for the 0.4-cos attention crossing). Default "all"
# keeps the falsified-global-sweep semantics; MU=0 keeps the gate bit-inert.
# Site is captured on ctx at FORWARD time (autograd runs the backward later).
# config.mm_prox_dirswap (env PTORCH_MM_PROX_DIRSWAP=1): keep the heuristic pipeline's per-tensor message
# AMPLITUDE (the golfed TD/floor/deflate bookkeeping) and swap only the
# DIRECTION to the Gram-damped adjoint. Scalar-freedom-legal: a per-tensor
# rescale of a better direction cannot re-introduce the amplitude-drift
# failure mode that killed raw uncapped adjoints (mm-uncapped 4.87), and it
# leaves the merge mixing amplitudes exactly as golfed. Motivated by the
# 2026-07-12 L2 decomposition probe: exact card A-side adjoints lift the
# attention-crossing demand cos 0.41->0.64 and the bottom block 0.09->0.47
# (interior exonerated, bit-identical under exact substitution).
_MM_PROX_SITE = "other"

# config.td_sibling_share (env PTORCH_TD_SIBLING_SHARE=1): ratio-preserving TD rescale across sibling cards
# (the q/k/v Linears of one attention module). Motivation (2026-07-12 L2
# probes): the TD floor path is the SOLE carrier of sublayer demand at trained
# equilibrium (TD off => path amplitude exactly 0), and its per-card
# `max(1, amp/r_norm)` rescale norms every sibling's message to the same
# amplitude — flattening the hopfield-consistent q:k:v ratios to 1:1:1
# (measured cost: crossing demand cos 0.41 vs 0.64 with exact ratios).
# Fix: rescale all siblings by ONE common factor amp·sqrt(n)/joint_norm using
# a one-step-stale joint norm (positive scalars: staleness safe, ratios
# preserved exactly within a step); see _td_apply. Default-off bit-inert.
_TD_SIBLING_GROUP = None


class td_sibling_group:
    """Context manager tagging MatMulProjection nodes as one TD sibling set."""

    def __init__(self, state):
        self.state = state

    def __enter__(self):
        global _TD_SIBLING_GROUP
        self._prev = _TD_SIBLING_GROUP
        _TD_SIBLING_GROUP = self.state
        return self

    def __exit__(self, *exc):
        global _TD_SIBLING_GROUP
        _TD_SIBLING_GROUP = self._prev
        return False


class mm_prox_site:
    """Context manager tagging MatMulProjection nodes created inside it."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        global _MM_PROX_SITE
        self._prev = _MM_PROX_SITE
        _MM_PROX_SITE = self.name
        return self

    def __exit__(self, *exc):
        global _MM_PROX_SITE
        _MM_PROX_SITE = self._prev
        return False


def td_bilinear_floor_update(A_det, A_ret, sibling=None, ctx=None, A=None, pairwise=False):
    """TD(λ) floor update at a bilinear solve, every scope: MatMulProjection,
    MatMulProjectionLinf and the fused attention nodes all call this.

    Norm mode (Variant B): blend the (optionally deflated) local residual norm
    into the floor, then rescale the activation target up to the floor (capped
    into the td_eps_lin linear-transport window). Only a norm crosses layers,
    so this is frame-invariant and width-agnostic. Graph scope reads the floor
    from this node's registry entry (`ctx`) and sends the updated floor to the
    producer of `A`; with no incoming floor the target stays purely local
    (== td_lambda 0 here). td_row (config.td_row) keeps one floor per token
    row; solves whose rows are not the token rows (attention-internal,
    `pairwise`) use the mean floor without updating it. Vector mode (Variant
    A) blends the residual vector itself. No-op until td_seed ran.
    """
    lam = config.td_lambda
    if lam == 0.0:
        return A_ret
    if config.td_mode != "norm":
        g = TD_TRACE["g"]
        if g is None:
            return A_ret
        # Bias is folded in as a padded ones column, so A may be one column
        # wider than the trace; the pad column's target is discarded by
        # F.pad's backward, so only the first d columns carry signal. Width
        # mismatch: leave the target purely local (no dimension bridge).
        d = g.shape[-1]
        r_local = A_det[..., :d] - A_ret[..., :d]
        if g.shape != r_local.shape:
            return A_ret
        g = (lam * g + (1.0 - lam) * r_local).detach()
        if config.td_clip > 0.0:
            g = g * (config.td_clip * TD_TRACE["seed_norm"] / g.norm().clamp_min(1e-12)).clamp(max=1.0)
        TD_TRACE["g"] = g
        if DIAG is not None:
            DIAG.record("td_trace", r_local_norm=float(r_local.norm()), g_norm=float(g.norm()))
        return torch.cat([A_det[..., :d] - g, A_ret[..., d:]], dim=-1)
    sibling = sibling if config.td_sibling_share else None
    if config.td_scope == "graph":
        floor = td_get(ctx)
        if floor is None:
            return A_ret
        A_ret, floor = _td_apply(A_det, A_ret, floor, update=True, sibling=sibling)
        td_send(A, floor)
        return A_ret
    if config.td_row:
        floor = TD_TRACE["floor_row"]
        if floor is None:
            return A_ret
        update = not pairwise and A_det.numel() // A_det.shape[-1] == floor.shape[0]
        A_ret, new = _td_apply(A_det, A_ret, floor if update else floor.mean(), update, rows=True)
        TD_TRACE["floor_row"] = new if update else floor
        return A_ret
    if TD_TRACE["floor"] is None:
        return A_ret
    A_ret, TD_TRACE["floor"] = _td_apply(A_det, A_ret, TD_TRACE["floor"], update=True, sibling=sibling)
    return A_ret
# ── end TD repeater ───────────────────────────────────────────────────────────


def _fold_bias(A, B, bias):
    """[A, 1] and [B; bias]: the bias as a ones column of the card's input."""
    return (A, B) if bias is None else (F.pad(A, (0, 1), value=1.0), torch.cat([B, bias]))


def _deflate_cap(r, x, td_eps_lin, dim=-1):
    """Strip the x-parallel self component of r per vector along `dim`, then
    cap its amplitude into the linear-transport window td_eps_lin·‖x‖."""
    r = r - (r * x).sum(dim, keepdim=True) / x.square().sum(dim, keepdim=True).clamp_min(1e-12) * x
    if td_eps_lin > 0.0:
        cap = td_eps_lin * x.norm(dim=dim, keepdim=True)
        r = r * (cap / r.norm(dim=dim, keepdim=True).clamp_min(1e-12)).clamp(max=1.0)
    return r


@torch.compile(dynamic=_TC_DYNAMIC, mode=_TC_MODE)
def matmul_proj_linf(A, B, Z, g=1.0, omega=1.0, num_steps=5, eps_init=None, legacy=False, direct_residual=False):
    """
    Exact independent bilinear projection for A @ B = Z using the L_infinity (Chebyshev) norm.

    Per (m, n) pair, solves for the smallest box radius eps with
    F(eps) = sum_k M_k(eps) + eps/g^2 - |Z - A@B| = 0 (paper, l_inf derivation;
    g plays gamma, as in the l2 solve). F is increasing but only piecewise
    convex (the opposite-sign branches are concave), so plain Newton can
    overshoot: the iterate is safeguarded inside the bracket [lo, hi] with
    F(lo) <= 0 <= F(hi), bisecting whenever a Newton step leaves it. One step
    from eps = 0 is the first-order (linearized) radius.
    eps_init: warm start for the radius (config.linf_warm_start), else 0.
    legacy: the pre-4de0227e9 step (config.linf_legacy_newton).
    direct_residual: return B - B_bar instead of B_bar, read straight off the
    proposals' mean displacement (see _matmul_proj_core).
    """
    M = A.size(-2)
    N = B.size(-1)

    # 1. Base pairwise projection
    P = A @ B  # (..., M, N)
    Z_target = Z * omega
    
    # Determine the direction of projection for each (m, n) pair
    # S = 1  => Case A (P < Z_target, need to increase P, decrease target)
    # S = -1 => Case B (P > Z_target, need to decrease P, increase target)
    S = torch.sign(Z_target - P) 
    D_abs = torch.abs(Z_target - P)
    eps = torch.zeros_like(P) if eps_init is None else eps_init  # the L_inf radius

    # 2. Precompute coordinate rotations (M, 1, K) and (1, N, K) 
    # Expanded implicitly to (M, N, K) during fused operations
    A_ext = A.unsqueeze(-1)       # (..., M, K, 1)
    B_ext = B.unsqueeze(-3)   # (..., 1, K, N)
    S_ext = S.unsqueeze(-2)       # (..., M, 1, N)
    
    U_sum = A_ext + B_ext
    V_diff = A_ext - B_ext
    
    u_abs = U_sum.abs() * 0.5
    v_abs = V_diff.abs() * 0.5
    
    # sign(0) = 1 (paper): a coordinate with a vanishing linear coefficient still
    # takes a corner of the box. torch.sign(0) = 0 would leave it in place.
    S_plus = torch.where(U_sum < 0, -1.0, 1.0).to(U_sum.dtype)
    S_minus = torch.where(V_diff < 0, -1.0, 1.0).to(V_diff.dtype)
    
    # Align u and v based on the geometric case (S)
    # S == 1  -> term_A tracks u_abs, term_B tracks v_abs
    # S == -1 -> term_A tracks v_abs, term_B tracks u_abs
    is_case_A = (S_ext == 1)
    
    base_A = torch.where(is_case_A, u_abs, v_abs)
    base_B = torch.where(is_case_A, v_abs, u_abs)
    
    # |dZ| <= eps/g^2 (gamma convention of the l2 solve), in omega-scaled Z units.
    target_penalty = omega / (g * g)
    damping = 1e-5

    # Bracket: F(0) = -D_abs <= 0, and F(D_abs/target_penalty) >= 0 since every M_k >= 0.
    lo = torch.zeros_like(P)
    hi = D_abs / target_penalty

    # 3. Safeguarded Newton for the optimal L_inf boundary (eps)
    for _ in range(num_steps):
        eps_ext = eps.unsqueeze(-2) # (..., M, 1, N)
        
        # Evaluate piecewise branches for the expanding L1 diamond
        term_A = 2.0 * eps_ext * base_A + eps_ext.square()
        term_B = 2.0 * eps_ext * base_B - eps_ext.square()
        
        # Determine which vertices of the diamond have crossed the axis:
        # term_A >= term_B ⟺ base_A + eps >= base_B for eps > 0; at the eps = 0
        # tie this picks the steeper branch, i.e. the right-derivative of the max.
        if legacy:
            wins_A = term_A >= term_B
        else:
            wins_A = base_A + eps_ext >= base_B # (..., M, K, N)
        
        # Reconstruct the objective function and derivatives piecewise
        max_terms = torch.where(wins_A, term_A, term_B)
        f_val = max_terms.sum(dim=-2) + eps * target_penalty - D_abs
        
        deriv_A = 2.0 * base_A + 2.0 * eps_ext
        deriv_B = 2.0 * base_B - 2.0 * eps_ext
        
        f_prime_k = torch.where(wins_A, deriv_A, deriv_B)
        f_prime_val = f_prime_k.sum(dim=-2) + target_penalty
        
        if legacy:  # damped Newton step, radius kept >= 0
            eps = torch.clamp(eps - f_val / (f_prime_val + damping), min=0.0)
            continue

        # F is increasing: F < 0 => root right of eps (raise lo), F > 0 => left (lower hi).
        lo = torch.where(f_val < 0, eps, lo)
        hi = torch.where(f_val > 0, eps, hi)

        # Damped Newton step, kept inside the closed bracket (else bisect).
        eps_newton = eps - f_val / (f_prime_val + damping)
        outside = (eps_newton < lo) | (eps_newton > hi)
        eps = torch.where(outside, 0.5 * (lo + hi), eps_newton)

    # 4. Analytical Consensus Reconstruction
    eps_ext = eps.unsqueeze(-2)
    
    # Re-evaluate winning branches with the highly accurate, final eps
    if legacy:
        wins_A = 2.0 * eps_ext * base_A + eps_ext.square() >= 2.0 * eps_ext * base_B - eps_ext.square()
    else:
        wins_A = base_A + eps_ext >= base_B

    # Route the delta updates based on the exact geometric branch taken
    dA_dir = torch.where(
        is_case_A,
        torch.where(wins_A, S_plus, -S_minus),
        torch.where(wins_A, S_minus, -S_plus)
    )
    
    dB_dir = torch.where(
        is_case_A,
        torch.where(wins_A, S_plus, S_minus),
        torch.where(wins_A, -S_minus, -S_plus)
    )
    
    dA = eps_ext * dA_dir
    dB = eps_ext * dB_dir
    
    # Average the independent point-wise proposals
    A_proj = A + dA.mean(dim=-1)
    B_proj = -dB.mean(dim=-3) if direct_residual else B + dB.mean(dim=-3)
    
    Z_proj = Z - S * (eps * target_penalty / omega)

    return A_proj, B_proj, Z_proj, eps.detach()


class MatMulProjectionLinf(torch.autograd.Function):
    """
    Minimize || (A_{new}, B_{new}, g^2 Z_{new}) - (A_{old}, B_{old}, g^2 Z_{old}) ||_inf
    subject to A_{new} @ B_{new} = Z_{new}  (per output pair, then consensus mean)
    A (1, N) `bias` joins the solve as a ones column of A (see MatMulProjection).
    B is always a weight: its gradient (and the bias's) is the residual
    theta - theta_bar, A's is its target.
    eps_cache: the layer's dict for config.linf_warm_start (None = cold start).
    """
    @staticmethod
    def forward(ctx, A, B, num_steps, g, omega, bias=None, eps_cache=None):
        out = (A @ B if bias is None else torch.addmm(bias, A.flatten(0, -2), B).view(*A.shape[:-1], -1)).div_(omega)
        ctx.save_for_backward(A, B, out, bias)
        ctx.g = g
        ctx.num_steps = num_steps
        ctx.omega = omega
        ctx.eps_cache = eps_cache
        return out

    @staticmethod
    def backward(ctx, Z_target):
        A, B, out, bias = ctx.saved_tensors
        A_det, B_det = _fold_bias(A.detach(), B.detach(), bias)
        Z_det = Z_raw = Z_target.detach()
        # two-channel stack hooks (mirror MatMulProjection): card-side global
        # consumption window on the SOLVE's Z; transit override below uses the
        # raw conserved target.
        if (config.transit_vjp and config.transit_vjp_card_cap
                and config.td_eps_lin > 0.0):
            dz = Z_det - out.detach()
            lim = config.td_eps_lin * out.detach().norm()
            Z_det = out.detach() + dz * (lim / dz.norm().clamp_min(1e-30)).clamp(max=1.0)

        # The L_inf solve materializes O(Mc*K*N) intermediates (~15 tensors);
        # stream row-chunks so each stays bounded (config.linf_chunk_elems caps
        # Mc*K*N). Rows are independent in the solve; B_proj recombines as the
        # row-count-weighted mean of the per-chunk proposals.
        A_2d = A_det.reshape(-1, A_det.shape[-1])
        Z_2d = Z_det.reshape(-1, Z_det.shape[-1])
        B_2d = (B_det.reshape(-1, B_det.shape[-2], B_det.shape[-1]).mean(dim=0)
                if B_det.ndim > 2 else B_det)
        M, K = A_2d.shape
        N = B_2d.shape[-1]
        cap = config.linf_chunk_elems
        mc = max(1, cap // max(K * N, 1))
        A_proj = torch.empty_like(A_2d)
        B_proj = torch.zeros_like(B_2d)
        # Warm start: the radius lives in one persistent (M, N) buffer, read and
        # overwritten in place, so a captured CUDA graph keeps carrying it.
        eps_buf = None
        if ctx.eps_cache is not None:
            eps_buf = ctx.eps_cache.get("eps")
            if eps_buf is None or eps_buf.shape != (M, N) or eps_buf.device != A_2d.device:
                eps_buf = ctx.eps_cache["eps"] = A_2d.new_zeros(M, N)
        for i in range(0, M, mc):
            A_proj[i:i + mc], B_p, _, eps = matmul_proj_linf(
                A_2d[i:i + mc], B_2d, Z_2d[i:i + mc],
                g=ctx.g, omega=ctx.omega, num_steps=ctx.num_steps,
                eps_init=None if eps_buf is None else eps_buf[i:i + mc],
                legacy=config.linf_legacy_newton, direct_residual=True)
            B_proj.add_(B_p, alpha=min(mc, M - i) / M)
            if eps_buf is not None:
                eps_buf[i:i + mc].copy_(eps)
        A_proj = A_proj.reshape(A_det.shape)
        B_proj = B_proj.expand(B_det.shape)

        A_ret = td_bilinear_floor_update(A_det, process_activation_target(A_det, A_proj), ctx=ctx, A=A)
        if config.matmul_transit_adjoint or config.transit_vjp or config.transit_vjp_matmul:
            s_z = (out.detach() - Z_raw) / ctx.omega
            r_a = s_z @ B_det.transpose(-2, -1)
            if not (config.transit_vjp or config.transit_vjp_matmul):
                r_a = _deflate_cap(r_a, A_det, config.td_eps_lin)
            A_ret = A_det - r_a
        if bias is None:
            return A_ret, B_proj, None, None, None, None, None
        return A_ret[..., :-1], B_proj[:-1], None, None, None, B_proj[-1:], None

@torch.compile(dynamic=_TC_DYNAMIC, mode=_TC_MODE)
def matmul_proj(A, B, Z, t_init=None, alpha=1.0, g=1.0, omega=1.0, num_steps=1, p_init=None, b_self_deflate=False, compute_z_proj=True, alpha_auto=False, out_cubic=0.0, direct_residual=False):
    """Exact bilinear projection for A @ B = Z; see _matmul_proj_core."""
    return _matmul_proj_core(A, B, Z, t_init, alpha, g, omega, num_steps, p_init,
                             b_self_deflate, compute_z_proj, alpha_auto, out_cubic,
                             direct_residual=direct_residual)[:4]


def _newton_t(p, q_eff, Z, t, alpha, target_penalty, out_cubic, num_steps):
    """The elementwise root-find of the bilinear projection: the t of each entry of
    p = A @ B, given q_eff = ||a_i||^2 + alpha*||b_j||^2. Only needs output-shaped
    tensors, so a conv can run it without forming its patch matrix."""
    # --- NUMERICAL SAFEGUARDS ---
    eps = 1e-5             # Minimum distance from the singularity
    max_step_size = 0.5    # Maximum allowable change in t per step

    # Calculate strict boundary for t to prevent alpha - t^2 from approaching 0
    # We require alpha - t^2 >= eps  =>  t^2 <= alpha - eps
    # (alpha may be a python float or a 0-dim tensor from auto-alpha.)
    if isinstance(alpha, torch.Tensor):
        max_t_val = (alpha - eps).clamp_min(eps).sqrt()
    else:
        max_t_val = math.sqrt(max(alpha - eps, eps))

    # 3. Safeguarded Newton (rtsafe) for the root of f(t) = 0.
    #
    # Per Elser, "Learning Without Loss" (arXiv:1911.00493), Lemma B.1: this root
    # equation (his eq. 52) is *strictly increasing* on the open interval t^2 < alpha,
    # sweeping -inf -> +inf, whenever q_eff > 2*sqrt(alpha)*|p|. That holds here by
    # Cauchy-Schwarz ( ||A||^2 + alpha*||B||^2 >= 2*sqrt(alpha)*|A.B| ), so the root is
    # unique and strictly *interior*. We bracket it in (-max_t_val, max_t_val) -- where
    # f(-max_t_val) < 0 < f(max_t_val) -- and fall back to bisection whenever a Newton
    # step would leave the bracket, so the iterate can never overshoot into the
    # singularity at t^2 = alpha (the boundary trap that made the old clamped Newton
    # blow up for far targets). A per-step movement clamp keeps the num_steps=1 path
    # bounded at |t| <= 0.5 (its previous behavior), and a convergence freeze keeps the
    # projection exactly idempotent when f(t) == 0.
    t = torch.clamp(t, min=-max_t_val, max=max_t_val)
    lo = torch.zeros_like(p) - max_t_val   # f(lo) < 0
    hi = torch.zeros_like(p) + max_t_val   # f(hi) > 0

    for _ in range(num_steps):
        t2 = t.square()
        alpha_minus_t2 = alpha - t2 # Guaranteed to be >= eps

        N_num = alpha * (p * (alpha + t2) + t * q_eff)
        # Output block: h̄⁺ = Z − target_penalty·(t + β·t³) (β = out_cubic).
        # β = 0 ⇒ t + 0·t³ = t, bit-identical to the linear map.
        f_val = (N_num / (alpha_minus_t2.square())) - Z + target_penalty * (t + out_cubic * t * t2)

        N_prime = alpha * (2.0 * t * p + q_eff)
        f_prime_val = ((N_prime * alpha_minus_t2) + 4.0 * t * N_num) / (alpha_minus_t2 ** 3) + target_penalty * (1.0 + 3.0 * out_cubic * t2)

        # f is strictly increasing: f>0 => root lies left of t (tighten hi), else lo.
        lo = torch.where(f_val < 0, t, lo)
        hi = torch.where(f_val > 0, t, hi)

        # Movement-limited Newton step (f_prime_val > 0 by monotonicity).
        step = torch.clamp(f_val / f_prime_val.clamp_min(1e-12), min=-max_step_size, max=max_step_size)
        t_newton = t - step

        # Accept Newton unless it leaves the CLOSED bracket; else bisect.
        # NaN fails both comparisons and propagates (surfaces upstream NaN bugs).
        # The interval must be closed: the bracket update above sets lo (or hi)
        # to the *current* t, and near the root the Newton step underflows fp32
        # (t_newton == t == lo). A strict test then declares the converged entry
        # "outside" and teleports it to the bracket midpoint — a t ~ sqrt(alpha)/2
        # excursion that needs ~30 bisection steps to undo, so every 1-5 step
        # solve returned garbage for exactly the entries that were already solved.
        outside = (t_newton < lo) | (t_newton > hi)
        t_next = torch.where(outside, 0.5 * (lo + hi), t_newton)

        # Freeze elements that have already hit the root (keeps idempotence
        # exact). |f| < 1e-10 alone is far too tight at fp32 scales (f is in
        # absolute Z units); an underflowed Newton step is the practical
        # fixed-point signal.
        converged = (f_val.abs() < 1e-10) | (t_newton == t)
        t = torch.where(converged, t, t_next)
    return t


def _matmul_proj_core(A, B, Z, t_init=None, alpha=1.0, g=1.0, omega=1.0, num_steps=1, p_init=None, b_self_deflate=False, compute_z_proj=True, alpha_auto=False, out_cubic=0.0, bias=None, compute_a_proj=True, direct_residual=False):
    """
    Exact bilinear projection for A @ B = Z.
    bias: optional (1, N) row solved as a ones column of A ([A, 1] @ [B; bias]),
    without materializing the padded operands. Its target is returned as a
    fifth output (None without bias); A_proj drops the ones column.
    p_init: optional precomputed A @ B (e.g. the saved forward product) to
    skip one matmul in the backward.

    out_cubic (β, config.proj_out_cubic; 0 = off, bit-identical): give the
    scalar OUTPUT block a non-quadratic (stiff-core / soft-tail) Bregman
    generator while keeping (h,θ) quadratic — so the scalar root-find is
    unchanged (the output is 1-D, its mirror map is O(1)). The linear output
    map h̄⁺ = Z − t/γ² becomes h̄⁺ = Z − (1/γ²)(t + β·t³): curvature is high near
    t=0 (stiff → amplifies small/deep residuals) and softens for large |t|
    (protects healthy top-layer residuals from over-inflation). f(t) stays
    strictly increasing for β ≥ 0, so the rtsafe bracket/Newton is unchanged.
    compute_z_proj: callers that discard Z_proj pass False to skip the
    full-size write. compute_a_proj=False likewise skips A_proj (returned as
    None) and its (M, N) @ (N, K) matmul, for an A that needs no target.
    direct_residual: return B - B_bar and bias - bias_bar instead of the
    targets, for a B that is a weight (what p.grad holds). First-layer residuals
    sit near the fp32 resolution of the weight, so subtracting two nearly equal
    tensors would lose much of them (17-46% measured). Uses alpha/(alpha-t^2) - 1
    = t^2/(alpha-t^2), so the self term never adds 1 to the coefficient.

    THEORY (2026-07-12, transport_theory.md §2, probe-verified): at small
    residual r = Z - A@B the closed form linearizes to a diagonally-
    preconditioned gradient step, ΔB = (1/M) A^T (r ⊘ D) with
    D = ||a_i||^2 + alpha*||b_j||^2 + alpha*omega^2/g^2 (cos 1.000000 to
    the code's output). The card solve is transport-agnostic — whether it is
    (preconditioned) backprop is decided entirely by what residual arrives.
    """
    M = A.size(-2)
    N = B.size(-1)


    # 1. Compute pairwise operations in O(M*N) without 3D expansion
    if p_init is not None:
        p = p_init  # (..., M, N)
    else:
        p = A @ B if bias is None else torch.addmm(bias, A, B)
    qa = (A * A).sum(dim=-1, keepdim=True)  # (..., M, 1)
    qb = (B * B).sum(dim=-2, keepdim=True)  # (..., 1, N)
    if bias is not None:  # the ones column and the bias row
        qa = qa + 1.0
        qb = qb + bias * bias
    if alpha_auto:
        # config.projection_alpha_auto: rescale so moving A and moving B cost
        # comparably, from the qa/qb reductions above (0-dim tensor, no sync;
        # power-of-two quantization kept for value-compatibility).
        ratio = (qa.mean() / qb.mean().clamp_min(1e-12)).clamp(1e-3, 1e6)
        alpha = alpha * torch.exp2(torch.round(torch.log2(ratio)))
    q_eff = qa + alpha * qb  # (..., M, N)

    # 2. Optional warm start for t
    t = torch.zeros_like(p) if t_init is None else t_init

    target_penalty = (omega ** 2) / (g ** 2)

    t = _newton_t(p, q_eff, Z, t, alpha, target_penalty, out_cubic, num_steps)

    # 4. Analytical Consensus Reconstruction
    t2 = t.square()
    denom = alpha - t2
    inv_denom = 1.0 / denom       # (M, N) - Safe because denom >= eps
    t_inv_denom = t / denom       # (M, N)

    # A_proj analytically averages over N proposals via Matrix Math
    A_proj = None
    if compute_a_proj:
        sum_inv_denom_j = inv_denom.sum(dim=-1, keepdim=True) # (..., M, 1)
        A_proj = (alpha / N) * (A * sum_inv_denom_j + t_inv_denom @ B.transpose(-2, -1))

    # B_proj analytically averages over M proposals via Matrix Math
    # With alpha/(alpha - t^2) - 1 = t^2/(alpha - t^2), the displacement never adds
    # 1 to the self coefficient. b_self_deflate drops that term, replacing the
    # coefficient (alpha/M)*sum_i 1/(alpha - t_ij^2) >= 1 by exactly 1: only the
    # transported adjoint stays (see config.matmul_b_self_deflate — the >=1
    # self-inflation never cancels and dominates the 1/M-diluted transported term
    # at GPT row counts).
    B_res = A.transpose(-2, -1) @ t_inv_denom
    bias_res = None if bias is None else t_inv_denom.sum(dim=-2, keepdim=True)
    if not b_self_deflate:
        self_coef = (t2 * inv_denom).sum(dim=-2, keepdim=True)  # (..., 1, N)
        B_res = B_res + B * self_coef
        bias_res = None if bias is None else bias_res + bias * self_coef
    B_res = (-1.0 / M) * B_res
    bias_res = None if bias is None else (-1.0 / M) * bias_res
    B_proj = B_res if direct_residual else B - B_res
    bias_proj = bias_res if direct_residual or bias is None else bias - bias_res

    Z_proj = (Z - target_penalty * (t + out_cubic * t * t2)) if compute_z_proj else None
    return A_proj, B_proj, Z_proj, t.detach(), bias_proj


def _card_proj_core(A, B, bias, out, Z, alpha, g, omega, num_steps: int, use_p_saved: bool,
                    b_self_deflate: bool, alpha_auto: bool, out_cubic: float, need_a: bool = True):
    """MatMulProjection's card solve: (..., K) rows of A
    against the K x N weight B, with the bias folded analytically. Returns A's
    target (None when not `need_a`, e.g. the first layer, whose input is data)
    and the residuals of B and bias (None without bias)."""
    A_s = A.reshape(-1, A.shape[-1])
    Z_s = Z.reshape(-1, Z.shape[-1]) * omega
    p = out.reshape(Z_s.shape) * omega if use_p_saved else None
    A_proj, B_proj, _, _, bias_proj = _matmul_proj_core(
        A_s, B, Z_s, None, alpha, g, omega, num_steps, p, b_self_deflate, False,
        alpha_auto, out_cubic, bias, need_a, direct_residual=True)
    return None if A_proj is None else A_proj.reshape(A.shape), B_proj, bias_proj


# One compiled call, so a card solve costs a single launch.
_card_proj = torch.compile(_card_proj_core, dynamic=_TC_DYNAMIC, mode=_TC_MODE)


# One compile per conv shape. The limit goes here, not in torch._dynamo.config: this runs
# in autograd's backward thread, which does not see config set on the main thread.
# Older torch has no such argument, and keeps the default limit.
_CONV_COMPILE_KWARGS = ({"recompile_limit": 16}
                        if "recompile_limit" in inspect.signature(torch.compile).parameters else {})


@torch.compile(dynamic=_TC_DYNAMIC, mode=_TC_MODE, **_CONV_COMPILE_KWARGS)
def _conv_card_proj(x, W, bias, out, Z, stride, padding, alpha, g, num_steps: int,
                    b_self_deflate: bool, alpha_auto: bool, out_cubic: float, need_x: bool):
    """_card_proj of a conv's patch matrix A followed by ConvPatchProjection's
    consensus, without forming A. x is NCHW, W is (O, C, kH, kW), out and Z are NCHW.
    With T = t/(alpha - t^2) and S = sum_j 1/(alpha - t^2):
        row norms of A      = box filter over sum_c x^2
        A^T T, fold(T W)    = the weight and input halves of the conv backward
        fold(A * S)         = x * (transposed box filter of S)
    Returns x's target (None when not `need_x`) and the residuals of W and bias."""
    O = W.shape[0]
    M = out.numel() // O
    conv_args = (None, stride, padding, [1, 1], False, [0, 0], 1)
    box = x.new_ones(1, 1, *W.shape[2:])

    qa = F.conv2d(x.square().sum(1, keepdim=True), box, stride=stride, padding=padding)
    qb = W.square().sum((1, 2, 3)).view(1, O, 1, 1)
    if bias is not None:
        qa = qa + 1.0
        qb = qb + bias.view(1, O, 1, 1).square()
    if alpha_auto:  # as in _matmul_proj_core
        ratio = (qa.mean() / qb.mean().clamp_min(1e-12)).clamp(1e-3, 1e6)
        alpha = alpha * torch.exp2(torch.round(torch.log2(ratio)))
    t = _newton_t(out, qa + alpha * qb, Z, torch.zeros_like(out), alpha, 1.0 / g ** 2, out_cubic, num_steps)

    t2 = t.square()
    inv_denom = 1.0 / (alpha - t2)
    t_inv_denom = t * inv_denom
    dx, dW, _ = torch.ops.aten.convolution_backward(t_inv_denom, x, W, *conv_args, [need_x, True, False])

    x_bar = None
    if need_x:
        def spread(v, like):  # transposed box filter: each output entry back onto its patch
            return torch.ops.aten.convolution_backward(v, like, box, *conv_args, [True, False, False])[0]
        count = spread(torch.ones_like(qa[:1]), x[:1, :1])  # patches holding each pixel
        x_bar = (alpha / O) * (x * spread(inv_denom.sum(1, keepdim=True), x[:, :1]) + dx)
        x_bar = torch.where(count > 0, x_bar / count, x)

    W_res = dW
    bias_res = None if bias is None else t_inv_denom.sum((0, 2, 3)).view(1, O)
    if not b_self_deflate:
        self_coef = (t2 * inv_denom).sum((0, 2, 3))
        W_res = W_res + W * self_coef.view(O, 1, 1, 1)
        bias_res = None if bias is None else bias_res + bias * self_coef
    return x_bar, (-1.0 / M) * W_res, None if bias is None else (-1.0 / M) * bias_res


# ─── autograd.Function wrappers ───────────────────────────────────────────────
def _plain_card(ctx, B):
    """True when a MatMulProjection backward is the plain card solve: a 2-D
    weight and none of the target overrides, TD or diagnostics on. It then runs
    as one compiled call (_card_proj), which launches far fewer kernels."""
    return not ctx.pairwise and B.ndim == 2 and plain_solve()


def plain_solve():
    """No target override, TD or diagnostics is on: a card solve is just the solve."""
    return (DIAG is None
            and config.td_lambda == 0.0 and not config.use_muon_activations
            and config.mm_prox_mu <= 0.0 and not config.matmul_transit_adjoint
            and not config.transit_vjp and not config.transit_vjp_matmul)


class MatMulProjection(torch.autograd.Function):
    """
    Minimize || A_{new} - A_{old} ||_F^2 + alpha * || B_{new} - B_{old} ||_F^2 + g * || Z_{new} - Z_{old} ||_F^2
    subject to A_{new} @ B_{new} = Z_{new}

    Linearized, the B-side update IS a diagonally-GN-preconditioned gradient
    step on the local misfit (ΔB = (1/M)A^T(r⊘D), cos 1.000000 verified) —
    see matmul_proj docstring and transport_theory.md §2.
    Every backward cold-starts t (rows of consecutive batches are
    unrelated) and Z_proj is not computed. A (1, N) `bias` (card solves only)
    joins the solve as a ones column of A, folded in by the backward.

    `pairwise` says what B is. False (a card solve, what Linear runs): B is a
    weight, and its gradient (and the bias's) is the residual theta - theta_bar
    that p.grad holds. True (attention's q k^T and P v): B is an activation, and
    its gradient is the target autograd passes on to the node below, like A's.
    """
    @staticmethod
    def forward(ctx, A, B, num_steps, alpha, g, omega, pairwise=False, value_solve=False, bias=None):
        # .grad of a weight holds a residual, but autograd passes targets between
        # nodes: a non-pairwise solve writes the residual, so B must be a weight.
        if not pairwise and (value_solve or not isinstance(B._base if B._is_view() else B, torch.nn.Parameter)):
            raise ValueError("a non-pairwise MatMulProjection needs a Parameter B; "
                             "pass pairwise=True for an activation B (and for value_solve)")
        ctx.alpha = alpha
        ctx.g = g
        ctx.num_steps = num_steps
        ctx.omega = omega
        ctx.pairwise = pairwise
        ctx.value_solve = value_solve
        ctx.mm_prox_site = _MM_PROX_SITE
        ctx.td_sibling = _TD_SIBLING_GROUP
        out = (A @ B if bias is None else torch.addmm(bias, A.flatten(0, -2), B).view(*A.shape[:-1], -1)).div_(omega)
        # Saving the output costs nothing (it stays alive as the next layer's
        # input) and lets the backward skip recomputing A @ B.
        ctx.save_for_backward(A, B, out, bias)
        return out

    @staticmethod
    def backward(ctx, Z_target):
        A, B, out, bias = ctx.saved_tensors
        if _plain_card(ctx, B):
            # The loss may already have run this solve with its own (_ce_card_proj).
            fused = ctx.__dict__.pop("fused_solve", None)
            A_ret, B_proj, bias_proj = fused if fused is not None else _card_proj(
                A.detach(), B.detach(), None if bias is None else bias.detach(), out.detach(),
                Z_target.detach(), ctx.alpha, ctx.g, ctx.omega, ctx.num_steps, config.p_saved,
                config.matmul_b_self_deflate, config.projection_alpha_auto, config.proj_out_cubic,
                ctx.needs_input_grad[0])
            # A_ret is consumed within the step; the weight/bias residuals may
            # outlive it as p.grad.
            return A_ret, _tc_own(B_proj), None, None, None, None, None, None, _tc_own(bias_proj)
        A_det, B_det = _fold_bias(A.detach(), B.detach(), bias)
        Z_det = Z_raw = Z_target.detach()
        if (not ctx.pairwise and config.transit_vjp
                and config.transit_vjp_card_cap and config.td_eps_lin > 0.0):
            # Card-side consumption window (see config): one global scalar so
            # within-tensor row/path weighting is untouched; the transit
            # override at the end reads the RAW Z_target, so the conserved
            # VJP chain is unaffected.
            dz = Z_det - out.detach()
            lim = config.td_eps_lin * out.detach().norm()
            Z_det = out.detach() + dz * (lim / dz.norm().clamp_min(1e-30)).clamp(max=1.0)

        if ctx.pairwise:
            A_s, B_s, Z_s = A_det, B_det, Z_det
        else:
            # Card solve: flatten (..., K) trunk rows against the K×N weight
            # (a batched weight is averaged first).
            A_s = A_det.reshape(-1, A_det.shape[-1])
            B_s = B_det.reshape(-1, *B_det.shape[-2:]).mean(0) if B_det.ndim > 2 else B_det
            Z_s = Z_det.reshape(-1, Z_det.shape[-1])
        # out*omega == A@B up to fp rounding; only valid where the solve uses
        # B as saved (the >2D non-pairwise path averages B first).
        p_saved = (out.detach() * ctx.omega).reshape(Z_s.shape) \
            if (config.p_saved and (ctx.pairwise or B_det.ndim == 2)) else None
        A_proj, B_proj, _, t_new = matmul_proj(
            A_s.contiguous(), B_s.contiguous(), Z_s * ctx.omega,
            alpha=ctx.alpha, g=ctx.g, omega=ctx.omega, num_steps=ctx.num_steps,
            p_init=p_saved, b_self_deflate=config.matmul_b_self_deflate,
            compute_z_proj=False, alpha_auto=config.projection_alpha_auto,
            out_cubic=config.proj_out_cubic, direct_residual=not ctx.pairwise,
        )
        A_proj = A_proj.reshape(A_det.shape)
        B_proj = B_proj.expand(B_det.shape)
        if DIAG is not None:
            t_abs = t_new.abs()
            key = ("matmul_pw_" if ctx.pairwise else "matmul_") + str(tuple(Z_det.shape))
            DIAG.record(
                key,
                t_mean=float(t_abs.mean()),
                t_max=float(t_abs.max()),
                t_sat=float((t_abs > 0.45).float().mean()),
                z_norm=float(Z_det.norm()),
                a_rel_move=float((A_proj - A_det).norm() / (A_det.norm() + 1e-12)),
            )
        del t_new  # only DIAG reads t: free the M×N buffer before the tail

        if (config.mm_prox_mu > 0.0 and not ctx.pairwise
                and (config.mm_prox_scope == "all"
                     or getattr(ctx, "mm_prox_site", "other") == config.mm_prox_scope)
                and A_det.ndim >= 2 and B_det.ndim == 2):
            # LM/damped-prox A-side transport override (see config.mm_prox_mu).
            # Solve is kept for B_proj; only the transported
            # activation target changes. Sign convention matches the
            # value-adjoint override: target = det + (message toward Z_target).
            # Batched inputs flatten rows against the shared K×K weight Gram.
            r_out = (Z_det - out.detach()) * ctx.omega          # (..., N)
            # K×K system in fp64: cond(I + μ'G) ≈ μ'σ²_max, and for K>N (or
            # anisotropic weights) G is (near-)rank-deficient — an fp32 solve
            # leaks O(cond·eps) null-space noise into the message.
            G = (B_det @ B_det.transpose(-2, -1)).double()      # (K, K)
            mu_eff = config.mm_prox_mu * G.shape[0] / G.diagonal().sum().clamp_min(1e-12)
            M_mat = torch.eye(G.shape[0], device=G.device, dtype=G.dtype) + mu_eff * G
            # M_mat is symmetric ⇒ dA = μ' r Bᵀ M⁻¹ = μ' r (M⁻¹B)ᵀ: solve against
            # the K×N weight instead of upcasting the M'×K right-hand side.
            W = (mu_eff * torch.linalg.solve(M_mat, B_det.double())).to(A_det.dtype)
            dA = r_out @ W.transpose(-2, -1)                    # (..., K)
            if config.mm_prox_dirswap:
                # per-tensor: heuristic amplitude × damped-adjoint direction
                n_h = (A_proj - A_det).norm()
                dA = dA * (n_h / dA.norm().clamp_min(1e-30))
            A_proj = A_det + dA

        if getattr(ctx, "value_solve", False) and config.attn_value_adjoint:
            # Attention PV solve (A=P softmax pattern (...,Tq,Tk), B=V (...,Tk,hd),
            # out=y=P@V): replace the joint solve's V target with the EXACT
            # backprop adjoint. The output-side residual (at the A@B scale) is
            # s_y = (out - Z_det)·omega and the exact adjoint w.r.t. V is
            #   δV = Pᵀ @ s_y = A.transpose(-2,-1) @ s_y  (dL/dV[j] = Σ_i P[i,j] s_y[i]).
            # Cap into the solver's fidelity window (td_eps_lin·‖V‖ per row) and
            # deflate the V-parallel self-component per row (td_deflate math).
            # This bypasses the joint solve's amplitude split, whose B-parallel
            # self-term dilutes the transported adjoint as N=T grows
            # (FINDINGS_attn_transport.md §4). B_proj is the direct V TARGET;
            # V_target = V_det − (capped, deflated) residual.
            s_y = (out.detach() - Z_det) * ctx.omega
            r_v = A_det.transpose(-2, -1) @ s_y            # Pᵀ s_y, shape of V
            # deflate the activation-parallel self-rescale artifact, cap into the window
            r_v = _deflate_cap(r_v, B_det, config.td_eps_lin)
            B_proj = B_det - r_v
            if DIAG is not None:
                DIAG.record(
                    "pv_adjoint" + str(tuple(B_det.shape)),
                    r_v_norm=float(r_v.norm()),
                    s_y_norm=float(s_y.norm()),
                )

        A_ret = process_activation_target(A_det, A_proj)
        if ctx.value_solve and config.td_lambda != 0.0 and config.attention_activation == "softmax":
            A_ret = torch.where(A_det != 0, A_ret, A_det)  # masked P entries carry no TD residual

        # TD(λ) floor update, every scope. The bias pad column is included in
        # the norm-mode rescale (F.pad's backward discards its target anyway).
        A_ret = td_bilinear_floor_update(A_det, A_ret, ctx.td_sibling, ctx, A, ctx.pairwise)

        # Attention demand-transit adjoints (config.attn_prob_adjoint /
        # config.attn_qk_adjoint, see config.py): replace the pairwise solves'
        # transit targets with the exact backprop adjoint, self-parallel
        # deflated and capped into the fidelity window — mirrors the
        # attn_value_adjoint treatment above. Overrides REPLACE the processed
        # A_ret/B_proj (incl. TD) for these operands; all other solves and the
        # default-off path are untouched.
        if ctx.pairwise:
            s_z = None
            if ((ctx.value_solve and config.attn_prob_adjoint)
                    or (not ctx.value_solve and config.attn_qk_adjoint)):
                # True adjoint in out units: out = A@B/ω ⇒ δA = (out−Z)@Bᵀ/ω
                # = s_z@Bᵀ/ω². Without the 1/ω² the QK hop (ω²=head_dim) lands
                # 64× hot and the fidelity cap pins every row — amplitude-flat
                # targets (observed: attnonly_adjt 4.168 vs baseline 4.031).
                s_z = (out.detach() - Z_det) / ctx.omega
                r_a = s_z @ B_det.transpose(-2, -1)
                if ctx.value_solve:  # masked / off-support P entries stay 0
                    r_a = r_a * (A_det != 0)
                A_ret = A_det - _deflate_cap(r_a, A_det, config.td_eps_lin)
            if not ctx.value_solve and config.attn_qk_adjoint:
                # B operand is kᵀ (..., hd, T): per-token vectors run along
                # dim -2, so deflate/cap over that axis.
                r_b = A_det.transpose(-2, -1) @ s_z
                B_proj = B_det - _deflate_cap(r_b, B_det, config.td_eps_lin, dim=-2)

        # Trunk demand-transit adjoint (config.matmul_transit_adjoint): the
        # two-channel split — weight cards keep their consensus targets
        # (B_proj untouched), but the demand handed DOWN the trunk (A_ret)
        # becomes the exact adjoint message, deflated and capped into the
        # fidelity window. Replaces the processed A_ret (incl. TD) for every
        # non-pairwise solve; default off = bit-identical.
        if not ctx.pairwise and (config.matmul_transit_adjoint
                                 or config.transit_vjp or config.transit_vjp_matmul):
            s_z = (out.detach() - Z_raw) / ctx.omega
            r_a = s_z @ B_det.transpose(-2, -1)
            if not (config.transit_vjp or config.transit_vjp_matmul):
                # capped variant: deflate + fidelity window (transit_vjp keeps
                # the raw adjoint — amplitude conservation is the point).
                r_a = _deflate_cap(r_a, A_det, config.td_eps_lin)
            A_ret = A_det - r_a

        if not ctx.pairwise:  # weight residual: may outlive the step as p.grad
            B_proj = _tc_own(B_proj)
        if bias is None:
            return A_ret, B_proj, None, None, None, None, None, None, None
        return A_ret[..., :-1], B_proj[:-1], None, None, None, None, None, None, B_proj[-1:]


class MSEProjection(torch.autograd.Function):
    """
    Minimize || predictions_{new} - predictions_{old} ||_2^2 + ||targets_{new} - targets_{old} ||_2^2 subject to 
    predictions_{new} = targets_{new}
    """
    @staticmethod
    def forward(ctx, predictions, targets):
        ctx.save_for_backward(predictions, targets)
        return F.mse_loss(predictions, targets)

    @staticmethod
    def backward(ctx, z):
        predictions, targets = ctx.saved_tensors
        out = (predictions + targets) / 2
        if config.td_lambda != 0.0:
            td_seed(predictions, (predictions - out).detach())
        return out, out

class CrossEntropyProjection(torch.autograd.Function):
    """
    prox_{lambda * l_CE(., y)}(x_0) = arg min_{x} (lambda * l_CE(x, y) + 1/2 * ||x - x_0||^2)
    """
    @staticmethod
    def forward(ctx, logits, labels, num_steps=5, lmbda = 1.0):
        ctx.save_for_backward(logits, labels)
        ctx.num_steps = num_steps
        ctx.lmbda = lmbda
        
        # The backward only needs the prox, so a trainer that never reads the loss
        # value can skip computing it (config.cross_entropy_loss_value).
        if not config.cross_entropy_loss_value:
            return logits.new_zeros(())
        # Standard cross entropy (hard or probability labels) as the scalar loss
        return F.cross_entropy(logits, labels)

    @staticmethod
    def backward(ctx, z_target):
        logits, labels = ctx.saved_tensors
        lmbda = ctx.lmbda
        steps = ctx.num_steps

        # A plain card head: one compiled call for the prox and the head's solve,
        # left on the head's node. The target handed to it is then unused.
        head = logits.grad_fn
        if isinstance(head, MatMulProjection._backward_cls):
            A, B, _, bias = head.saved_tensors
            if _plain_card(head, B):
                head.fused_solve = _ce_card_proj(
                    logits.detach(), labels, steps, lmbda, A.detach(), B.detach(),
                    None if bias is None else bias.detach(), head.alpha, head.g, head.omega,
                    head.num_steps, config.p_saved, config.matmul_b_self_deflate,
                    config.projection_alpha_auto, config.proj_out_cubic, head.needs_input_grad[0])
                return logits, None, None, None

        x = _ce_prox_impl(logits, labels, steps, lmbda)

        if config.td_lambda != 0.0:
            td_seed(logits, (logits - x).detach())

        return x, None, None, None

class PolicyGradientProjection(torch.autograd.Function):
    """
    Projection-based REINFORCE loss (replaces Categorical(logits=...).log_prob).

    Forward returns the usual pseudo-loss -(logp * weights).mean() for logging.
    Backward ignores the incoming grad and returns a *target* for the logits:
    a proximal / gradient-ascent step on  w * logp(a | logits),
        logits <- logits + lambda * w * (onehot(a) - softmax(logits)),
    iterated num_steps times. Mirrors CrossEntropyProjection, with the sampled
    action as the target and the per-sample weight scaling the step.
    """
    @staticmethod
    def forward(ctx, logits, act, weights, num_steps=5, lmbda=1.0):
        ctx.save_for_backward(logits, act, weights)
        ctx.num_steps = num_steps
        ctx.lmbda = lmbda
        logp_all = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        logp = logp_all.gather(-1, act.long().unsqueeze(-1)).squeeze(-1)
        return -(logp * weights).mean()

    @staticmethod
    def backward(ctx, z):
        logits, act, weights = ctx.saved_tensors
        onehot = F.one_hot(act.long(), num_classes=logits.size(-1)).to(logits.dtype)
        w = weights.unsqueeze(-1)                       # (B, 1)
        x = logits
        for _ in range(ctx.num_steps):
            x = x + ctx.lmbda * w * (onehot - F.softmax(x, dim=-1))
        return x, None, None, None, None                # target logits


class GaussianPolicyGradientProjection(torch.autograd.Function):
    """
    Continuous-action analog of PolicyGradientProjection: a diagonal Gaussian
    policy a ~ N(mu, sigma) with sigma = exp(log_std) shared across the batch.

    Forward returns the REINFORCE pseudo-loss -(logp * weights).mean() for
    logging. Backward ignores the incoming grad and takes one gradient-ascent
    step on  w * logp(a | mu, log_std), iterated num_steps times:

        d logp / d mu       = (a - mu) / sigma^2
        d logp / d log_std  = ((a - mu) / sigma)^2 - 1

        mu       <- mu       + lambda * w * (a - mu) / sigma^2
        log_std  <- log_std  + lambda * mean_B[ w * (((a-mu)/sigma)^2 - 1) ]

    mu_target is per-sample (B, A) and flows back through the projection layers
    like the discrete logits target. log_std is a single shared (A,) parameter,
    so its per-sample updates are reduced over the batch (mirroring the .mean()
    in the forward loss). log_std is a parameter, so it gets the residual
    log_std - log_std_target (the accumulated move, negated) in log_std.grad;
    mu is an activation and gets its target. The log_std update is unbounded
    (unlike onehot - softmax), so log_std is clamped to keep sigma sane.

    The mu step carries a 1/sigma^2 factor: as sigma shrinks it explodes, so the
    target runs away and the projection layers diverge. ``max_delta`` clamps the
    per-step mu move element-wise, which removes that blow-up *at the source* --
    with it, sigma can be driven low (a committed, low-noise gait) without the
    mu target detonating, so the log_std floor no longer has to be propped up
    high purely for stability.
    """
    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    @staticmethod
    def forward(ctx, mu, log_std, act, weights, num_steps=5, lmbda=1.0, max_delta=None):
        ctx.save_for_backward(mu, log_std, act, weights)
        ctx.num_steps = num_steps
        ctx.lmbda = lmbda
        ctx.max_delta = max_delta
        std = torch.exp(log_std)
        logp = -0.5 * (
            ((act - mu) / std) ** 2 + 2.0 * log_std + math.log(2.0 * math.pi)
        ).sum(-1)
        return -(logp * weights).mean()

    @staticmethod
    def backward(ctx, z):
        mu, log_std, act, weights = ctx.saved_tensors
        w = weights.unsqueeze(-1)                       # (B, 1)
        cls = GaussianPolicyGradientProjection
        mu_t, d_ls = mu, torch.zeros_like(log_std)  # d_ls: log_std's total move
        for _ in range(ctx.num_steps):
            std = torch.exp(log_std + d_ls)
            delta = ctx.lmbda * w * (act - mu_t) / std ** 2
            if ctx.max_delta is not None:
                delta = delta.clamp(-ctx.max_delta, ctx.max_delta)
            d_ls = d_ls + ctx.lmbda * (w * (((act - mu_t) / std) ** 2 - 1.0)).mean(0)
            mu_t = mu_t + delta
            d_ls = torch.minimum(torch.maximum(d_ls, cls.LOG_STD_MIN - log_std),
                                 cls.LOG_STD_MAX - log_std)  # log_std + d_ls stays in range
        # mu target, log_std residual; trailing Nones match the extra fwd args
        return mu_t, -d_ls, None, None, None, None, None

class HardMarginProjection(torch.autograd.Function):
    """
    Your original implementation: The 'Strict Teacher'.
    Hard-clips logits to the boundary immediately. 
    """
    @staticmethod
    def forward(ctx, logits, labels):
        ctx.save_for_backward(logits, labels)
        return logits

    @staticmethod
    def backward(ctx, z_target):
        logits, labels = ctx.saved_tensors
        return _hard_margin_impl(logits, labels), None


@torch.compile(dynamic=_TC_DYNAMIC, mode=_TC_MODE)
def _hard_margin_impl(logits, labels):
    # torch.where, not boolean indexing: indexing runs nonzero, which syncs with the GPU.
    cond_0 = (labels <= 0) & (logits > 0)
    cond_1 = (labels > 0) & (logits < labels)
    return torch.where(cond_0, torch.zeros_like(logits), torch.where(cond_1, labels, logits))


class ProximalHingeMargin(torch.autograd.Function):
    """
    Soft Margin Variant 1: The 'Hinge' Proximal Operator.
    Instead of teleporting all the way to the boundary, it takes a bounded step 
    (size lambda_val) towards the correct margin.
    """
    @staticmethod
    def forward(ctx, logits, labels, lambda_val=1.0):
        ctx.save_for_backward(logits, labels)
        ctx.lambda_val = lambda_val
        return logits

    @staticmethod
    def backward(ctx, z_target):
        logits, labels = ctx.saved_tensors
        lmbda = ctx.lambda_val

        cond_0 = (labels <= 0) & (logits > 0)
        cond_1 = (labels > 0) & (logits < labels)

        new_logits = logits.clone()
        
        new_logits[cond_0] = torch.maximum(logits[cond_0] - lmbda, torch.zeros_like(logits[cond_0]))
        
        # Move towards the label by at most lambda, but don't overshoot the label
        new_logits[cond_1] = torch.minimum(logits[cond_1] + lmbda, labels[cond_1])

        return new_logits, None, None


class EmbeddingProjection(torch.autograd.Function):
    """Projection-based token embedding (row-gather as a fixed one-hot bilinear).

    An embedding lookup is ``onehot(ids) @ W`` — a bilinear form whose left
    operand is a fixed one-hot, so unlike MatMulProjection there is nothing to
    solve for on the left. The backward receives the target ``X̄`` for the
    gathered rows (shape ``(..., dim)``) and returns the residual of the exact
    min-norm projection of ``W``: for each vocab row ``v`` selected by some id,
    the negated mean of the displacements ``X̄ - W[v]`` over the positions where
    ``ids == v`` (scatter-mean via ``index_add_``). Rows that no id selected get
    a zero residual, exactly like an untouched B row of MatMulProjection.

    Weight tying: a weight shared with an lm_head also receives the head's
    residual through autograd, and residuals are additive, so ``.grad`` simply
    holds their sum. That sum is rarely what training wants, though: the two
    are not on a common scale (the head's shrinks with the batch, the
    embedding's is an O(1) activation displacement, ~1000x larger), so it is
    the embedding pull alone. To let the training loop weigh them, a ``sink``
    (1-element list, set by the tied Embedding module) receives the embedding's
    residual in ``sink[0]`` and ``None`` is returned for the weight, so
    ``.grad`` holds the head's alone (see accumulate_step).

    Terminal TD consumer: in graph scope / norm mode it pops (drains) any floor
    routed to this node so it does not leak, but — being terminal — sends
    nothing onward (no td_send).
    """
    @staticmethod
    def forward(ctx, weight, ids, sink=None):
        ctx.save_for_backward(weight, ids)
        ctx.vocab = weight.shape[0]
        ctx.tied = sink is not None
        ctx.sink = sink
        return F.embedding(ids, weight)

    @staticmethod
    def backward(ctx, x_bar):
        weight, ids = ctx.saved_tensors
        # Terminal consumer: drain any TD floor routed here (graph + norm only),
        # matching Conversion/MatMulProjection so floors do not leak past us.
        td_get(ctx)
        w_det = weight.detach()
        dim = w_det.shape[-1]
        x_flat = x_bar.reshape(-1, dim)
        ids_flat = ids.reshape(-1)
        # scatter-sum of the row DISPLACEMENTS over the vocab (summing the
        # absolute targets, each ~w, cancels catastrophically in fp32); rows
        # no id selected get zero displacement.
        w_f = w_det.to(x_flat.dtype)
        disp = torch.zeros_like(w_f).index_add_(0, ids_flat, x_flat - w_f[ids_flat])
        if config.transit_vjp or config.transit_vjp_embed:
            # Conservative scatter-SUM of displacements (gradient semantics):
            # the row moves by sum_pos (x_bar_pos - w_v). The scatter-mean is one
            # more consensus artifact — it under-weights frequent tokens by
            # their occurrence count relative to the true gradient (tok_emb
            # was the last cos-0.60 family when every card sat at 0.95+).
            residual = (-disp).to(w_det.dtype)
        else:
            counts = torch.zeros_like(w_f[:, 0]).index_add_(0, ids_flat, torch.ones_like(x_flat[:, 0]))
            # count^beta consensus reweighting of the mean displacement (see
            # config.embed_count_beta): beta=1 recovers scatter-sum demand,
            # beta=0 the scatter-mean.
            scale = counts.clamp_min(1.0).pow(config.embed_count_beta - 1.0).unsqueeze(1)
            residual = (-disp * scale).to(w_det.dtype)
        if ctx.tied:
            ctx.sink[0] = residual
            return None, None, None
        return residual, None, None

class ReLULInfinityProjection(torch.autograd.Function):
    """
    Sets the new input directly based on the target z,
    projecting onto the ReLU graph using the L-infinity norm.
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.relu(x)

    @staticmethod
    def backward(ctx, z):
        x, = ctx.saved_tensors
        return process_node_target(x, _relu_linf_impl(x, z), ctx), None


@torch.compile(dynamic=_TC_DYNAMIC, mode=_TC_MODE)
def _relu_linf_impl(x, z):
    # Solution 1: project onto inactive branch (x <= 0, output = 0)
    x_1 = torch.clamp(x, max=0)
    # L-infinity distance: max(|x - x_1|, |z - 0|)
    dist_1 = torch.max(torch.abs(x - x_1), torch.abs(z))

    # Solution 2: project onto active branch (x > 0, output = x)
    x_2 = torch.clamp((x + z) / 2.0, min=0)
    # L-infinity distance: max(|x - x_2|, |z - x_2|)
    dist_2 = torch.max(torch.abs(x - x_2), torch.abs(z - x_2))

    # Select solution minimizing L-infinity distance
    return torch.where(dist_1 < dist_2, x_1, x_2)
    
class ReLUProjection(torch.autograd.Function):
    """
    Sets the new input directly based on the target z.
    Ignores the original input x entirely.
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.relu(x)

    @staticmethod
    def backward(ctx, z):
        x, = ctx.saved_tensors
        return process_node_target(x, _relu_l2_impl(x, z), ctx), None


@torch.compile(dynamic=_TC_DYNAMIC, mode=_TC_MODE)
def _relu_l2_impl(x, z):
    # Solution 1: project onto inactive branch (x <= 0, output = 0)
    x_1 = torch.clamp(x, max=0)
    dist_1 = (x - x_1)**2 + z**2

    # Solution 2: project onto active branch (x > 0, output = x)
    x_2 = torch.clamp((x + z) / 2.0, min=0)
    dist_2 = (x - x_2)**2 + (z - x_2)**2

    # Select solution minimizing distance
    return torch.where(dist_1 < dist_2, x_1, x_2)


def _project_simplex_sorted(z):
    """Euclidean projection of each row of z onto the probability simplex.

    Sort-based algorithm (Held et al. 1974 / Duchi et al. 2008); rows are the
    last dimension. Cheap for attention-sized rows (T ~ 64).
    """
    u, _ = torch.sort(z, dim=-1, descending=True)
    css = torch.cumsum(u, dim=-1) - 1.0
    j = torch.arange(1, z.size(-1) + 1, device=z.device, dtype=z.dtype)
    rho = (u - css / j > 0).to(z.dtype).sum(dim=-1, keepdim=True).clamp(min=1)
    tau = torch.gather(css, -1, rho.long() - 1) / rho
    return torch.clamp(z - tau, min=0.0)


# ── compiled backward impls (pure tensor arithmetic; config/DIAG stay outside) ─
@torch.compile(dynamic=_TC_DYNAMIC, mode=_TC_MODE)
def _softmax_relfloor_impl(x, z, eps_rel: float, kappa: float, clip: float, eps: float):
    mask = (x != float('-inf'))
    safe_x = torch.where(mask, x, torch.zeros_like(x))
    p = F.softmax(x, dim=-1)
    z_pos = torch.maximum(z, eps_rel * p) * mask
    z_bar = z_pos / z_pos.sum(dim=-1, keepdim=True).clamp_min(eps)
    log_z = torch.log(z_bar.clamp_min(eps))
    diff = (safe_x - log_z) * mask
    c = diff.sum(dim=-1, keepdim=True) / mask.sum(dim=-1, keepdim=True).clamp(min=1)
    step = torch.clamp(log_z + c - safe_x, min=-clip, max=clip) * mask
    return safe_x + kappa * step, z_bar, safe_x


@torch.compile(dynamic=_TC_DYNAMIC, mode=_TC_MODE)
def _hardtanh_impl(x, z, a: float):
    inv_a = 1.0 / a
    x_1 = torch.clamp(x, max=-inv_a)
    dist_1 = (x - x_1) ** 2 + (z + 1.0) ** 2
    x_2 = torch.clamp((x + a * z) / (1.0 + a * a), min=-inv_a, max=inv_a)
    dist_2 = (x - x_2) ** 2 + (z - a * x_2) ** 2
    x_3 = torch.clamp(x, min=inv_a)
    dist_3 = (x - x_3) ** 2 + (z - 1.0) ** 2
    result = torch.where(dist_2 <= dist_1, x_2, x_1)
    best = torch.minimum(dist_1, dist_2)
    return torch.where(dist_3 < best, x_3, result)


@torch.compile(dynamic=_TC_DYNAMIC, mode=_TC_MODE)
def _leakyrelu_impl(x, z, slope: float):
    x_1 = torch.clamp((x + slope * z) / (1.0 + slope * slope), max=0)
    dist_1 = (x - x_1) ** 2 + (z - slope * x_1) ** 2
    x_2 = torch.clamp((x + z) / 2.0, min=0)
    dist_2 = (x - x_2) ** 2 + (z - x_2) ** 2
    return torch.where(dist_1 < dist_2, x_1, x_2)


@torch.compile(dynamic=_TC_DYNAMIC, mode=_TC_MODE)
def _leakyrelu_linf_impl(x, z, slope: float):
    # Chebyshev projection onto the branch line y = s*t equalizes the two
    # coordinate errors |x - t| = |z - s*t|  ->  t = (x + z)/(1 + s); the L2
    # projection (x + s*z)/(1 + s^2) instead scales the transported demand by
    # the branch slope (~s*z for s << 1) — the gradient's sensitivity scaling,
    # a vanishing-targets dissipation channel on the flat branch.
    x_1 = torch.clamp((x + z) / (1.0 + slope), max=0)
    dist_1 = torch.max(torch.abs(x - x_1), torch.abs(z - slope * x_1))
    x_2 = torch.clamp((x + z) / 2.0, min=0)
    dist_2 = torch.max(torch.abs(x - x_2), torch.abs(z - x_2))
    return torch.where(dist_1 < dist_2, x_1, x_2)


@torch.compile(dynamic=_TC_DYNAMIC, mode=_TC_MODE)
def _rmsnorm_legacy_impl(x, z_target):
    n = x.size(-1)
    z_target_norm = torch.linalg.norm(z_target, dim=-1, keepdim=True)
    z_bar = math.sqrt(n) * (z_target / (z_target_norm + 1e-8))
    sigma_bar = (x * z_bar).mean(dim=-1, keepdim=True)
    return sigma_bar * z_bar, sigma_bar


@torch.compile(dynamic=_TC_DYNAMIC, mode=_TC_MODE)
def _hadamard_impl(a, b, z_target, eps: float):
    # b-target: L2 projection of (b, z_target) onto z = a*b at fixed a (the
    # Hardtanh/LeakyReLU template). Idempotent, and the b move a*r/(1+a^2)
    # stays bounded as a crosses zero, where the old midpoint (b + z/a)/2
    # amplified the delta by 1/(2|a|).
    b_bar = b + a * (z_target - a * b) / (1.0 + a * a)
    # a-residual a - a_bar, a_bar = (a + (z.b) / s) / 2 the midpoint with the
    # least-squares fit, s = max(b.b, eps). Over a common denominator
    # a - (z.b)/s = (b.(a b - z) + a (s - b.b)) / s, whose numerator is built
    # from the output misfit rather than from two nearly equal a-sized terms.
    b_sq_sum = (b * b).reshape(-1, b.shape[-1]).sum(dim=0)
    s = b_sq_sum.clamp(min=eps)
    misfit = (b * (a * b - z_target)).reshape(-1, b.shape[-1]).sum(dim=0)
    return (misfit + a * (s - b_sq_sum)) / (2 * s), b_bar


def _ce_prox_core(x0, y, num_steps: int, lmbda: float):
    # Fixed point of x = x0 + lmbda*(y - |y|_1 softmax(x)) (contraction for
    # lmbda < 2). Hard labels become a fused one-hot; ignore_index rows match
    # no class, so y = 0 there and the row keeps x0.
    if y.ndim < x0.ndim:
        y = (torch.arange(x0.size(-1), device=x0.device) == y.unsqueeze(-1)).to(x0.dtype)
    # torch.sum, not y.sum: dynamo cannot trace the Tensor.sum override's
    # captured TensorBase.sum and graph-breaks there (torch.sum traces).
    s = torch.sum(y, dim=-1, keepdim=True)
    x = x0
    for _ in range(num_steps):
        x = x0 + lmbda * (y - s * F.softmax(x, dim=-1))
    return x


_ce_prox_impl = torch.compile(_ce_prox_core, dynamic=_TC_DYNAMIC, mode=_TC_MODE)


@torch.compile(dynamic=_TC_DYNAMIC, mode=_TC_MODE)
def _ce_card_proj(logits, y, ce_steps: int, lmbda: float, A, B, bias, alpha, g, omega, num_steps: int,
                  use_p_saved: bool, b_self_deflate: bool, alpha_auto: bool, out_cubic: float,
                  need_a: bool = True):
    """The CE prox and the solve of the card that produced the logits, in one call."""
    return _card_proj_core(A, B, bias, logits, _ce_prox_core(logits, y, ce_steps, lmbda), alpha, g,
                           omega, num_steps, use_p_saved, b_self_deflate, alpha_auto, out_cubic, need_a)
# ── end compiled backward impls ────────────────────────────────────────────────


class SoftmaxProjection(torch.autograd.Function):
    """
    Softmax projection via mixed L2/KL geometry.

    Backward modes (config.softmax_target_mode):
      "legacy":     clamp(z, 1e-8) + L1 renormalize + exact log-shift. A
                    slightly negative consensus target entry becomes a ~-18.4
                    log-target vs logits of O(+-3), injecting huge logit
                    displacements upstream.
      "rel_floor":  floor the target relative to the *current* softmax output
                    (z >= eps_rel * p), renormalize, then a damped and clipped
                    log-space step. Bounds per-entry displacement.
      "simplex_l2": Euclidean (sort-based) projection onto the simplex first,
                    then the same rel_floor + damped log-space step (exact
                    zeros from the simplex projection cannot be logged).
      "jacobian":   exact softmax VJP (see backward). CONFIRMED KEEPER
                    (2026-07-12, STATUS_LEDGER.md): in every winning stack.
                    The log-space modes above are FALSIFIED for attention
                    transport — they deliver cos~0.08 logit targets (preimage
                    1/sigma transport at the softmax bottleneck,
                    transport_theory.md §3 Corollary B), killing q/k learning.
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return F.softmax(x, dim=-1)

    @staticmethod
    def backward(ctx, z):
        x, = ctx.saved_tensors
        eps = 1e-8
        mode = config.softmax_target_mode
        # Masked (-inf, e.g. causal) logits: every mode works on the finite
        # logits safe_x plus the visibility mask, so no -inf reaches the
        # generic (x, x_bar) post-processing of process_node_target (r = x -
        # x_bar would be NaN there). Masked entries get the placeholder target
        # 0 (zero residual), which MaskedAddProjection replaces with its input.

        # Fast path: fused rel_floor backward (the (B,H,T,T) elementwise chain
        # dominates attention backward cost when run eagerly). DIAG runs keep
        # the eager path for its intermediates.
        if mode == "rel_floor" and DIAG is None and config.fused_softmax:
            x_bar, z_bar, safe_x = _softmax_relfloor_impl(
                x, z, config.softmax_target_eps_rel,
                config.softmax_target_kappa, config.softmax_logit_clip, eps,
            )
            return process_node_target(safe_x, x_bar, ctx), None

        mask = (x != float('-inf'))
        safe_x = torch.where(mask, x, torch.zeros_like(x))

        # Jacobian-adjoint mode: the log-space simplex projections ("legacy",
        # "rel_floor", "simplex_l2") deliver a well-aligned target at the softmax
        # OUTPUT but a logit target decorrelated from the true softmax-Jacobian
        # adjoint (measured cos ~0.08 vs the reference, vs 0.95 at the output —
        # see experiments/attention/FINDINGS_attn_transport.md). Downstream the
        # QK bilinear solve then faithfully transports that garbage into q,k
        # (cos ~0.05), annihilating the query/key learning pathway in attention.
        #
        # Here we instead return the exact softmax vector-Jacobian product, i.e.
        # what backprop delivers: with output residual s = p - z (analog of a
        # grad_output), the logit residual is s_logit = p ⊙ (s − Σ_j p_j s_j)
        # and the logit target is x − s_logit. Direction-faithful by construction
        # (magnitude is restored by Muon / the TD repeater downstream). Masked
        # entries carry p = 0 so contribute nothing. Same compiled kernel as the
        # fused attention node.
        if mode == "jacobian":
            from .attention_fused import _softmax_mask_jacobian_impl
            x_bar = _softmax_mask_jacobian_impl(safe_x, z, F.softmax(x, dim=-1), mask)
        elif mode == "legacy":
            # Step 1: positivity + KL projection onto the simplex (L1 normalization).
            z_pos = torch.clamp(z, min=eps)
            z_bar = z_pos / z_pos.sum(dim=-1, keepdim=True)

            # Step 2: shift log(z_bar) to be closest in L2 to x along the
            # softmax-equivalence ray.
            log_z = torch.log(z_bar)
            diff = (safe_x - log_z) * mask
            c = diff.sum(dim=-1, keepdim=True) / mask.sum(dim=-1, keepdim=True).clamp(min=1)
            x_bar = torch.where(mask, log_z + c, safe_x)
        else:
            p = F.softmax(x, dim=-1)  # -inf entries get exactly 0 weight

            z_work = z
            if mode == "simplex_l2":
                # Push masked entries far negative so the simplex projection
                # assigns them 0 instead of stealing mass.
                z_work = z.masked_fill(~mask, -1e30)
                z_work = _project_simplex_sorted(z_work)

            # Relative floor: never let a target entry fall below eps_rel of
            # the current weight, so log(z_bar) - log(p) >= ~log(eps_rel).
            z_pos = torch.maximum(z_work, config.softmax_target_eps_rel * p)
            z_pos = z_pos * mask
            z_bar = z_pos / z_pos.sum(dim=-1, keepdim=True).clamp_min(eps)

            # Damped, clipped log-space step along the softmax-equivalence ray.
            log_z = torch.log(z_bar.clamp_min(eps))
            diff = (safe_x - log_z) * mask
            c = diff.sum(dim=-1, keepdim=True) / mask.sum(dim=-1, keepdim=True).clamp(min=1)
            clip = config.softmax_logit_clip
            step = torch.clamp(log_z + c - safe_x, min=-clip, max=clip) * mask
            x_bar = safe_x + config.softmax_target_kappa * step

        if DIAG is not None:
            disp = (x_bar - safe_x).abs()  # zero at masked entries
            DIAG.record(
                "softmax",
                clamp_frac=0.0 if mode == "jacobian" else float(((z < eps) & mask).float().mean()),
                z_min=float(z.min()),
                max_logit_disp=float(disp.max()),
                mean_logit_disp=float(disp.mean()),
            )

        return process_node_target(safe_x, x_bar, ctx), None

class MaskedAddProjection(torch.autograd.Function):
    """
    Projection-aware addition for causal masks.
    Forward: x + mask
    Backward: Replaces the target for masked positions with the original x, 
              so the pseudo-gradient is zero for masked positions.
    """
    @staticmethod
    def forward(ctx, x, mask):
        ctx.save_for_backward(x, mask)
        return x + mask

    @staticmethod
    def backward(ctx, target_out):
        x, mask = ctx.saved_tensors
        # Undo the additive mask; where it is -inf, keep x
        target_x = torch.where(mask == float('-inf'), x, target_out - mask)
        return process_node_target(x, target_x, ctx), None

class BranchProjection(torch.autograd.Function):
    """
    Projection-aware branching for fan-out points.

    `Branch` creates one BranchProjection node per consumer; autograd *sums*
    the per-node backward outputs into the shared input. Invariant: each node
    must have exactly one consumer (guaranteed by Branch.forward) — reusing a
    single branch output twice would silently double-count its contribution.

    Modes (resolved by the Branch module from config.branch_mode):
      "mean":      each node returns t_i / N, so the accumulated target is the
                   consensus average (1/N) * sum_i t_i (legacy). In residual
                   nets this halves the skip-path delta at every sublayer
                   split — the vanishing-targets mechanism.
      "delta_sum": each node returns t_i - ((N-1)/N) * x; the autograd sum
                   reproduces x_bar = x + sum_i (t_i - x) exactly — the
                   backprop fan-out analog, keeping gain-1 skip paths.

    STATUS (2026-07-12): both combines are METRIC-BLIND — no combine rule can
    fix branch messages whose accumulated path scalars disagree (probe P3:
    merged cos to gradient 0.044 for mean and delta_sum alike). The practical
    fix is trainer-level MERGE_RHO fan-in equalization (skip = gain-1
    amplitude reference, sublayer displacement rescaled to c_hat*||skip||) —
    transport_theory.md axiom T2; −0.19 nats, no-VJP record 3.2465.
    """
    @staticmethod
    def forward(ctx, x, num_branches, mode="mean"):
        ctx.num_branches = float(num_branches)
        ctx.mode = mode
        # Graph scope needs x's producer to forward the floor to; "mean" mode
        # would not otherwise save x.
        _td_graph = (config.td_lambda != 0.0 and config.td_mode == "norm"
                     and config.td_scope == "graph")
        if mode == "delta_sum" or _td_graph:
            ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        n = ctx.num_branches
        # Graph scope: forward each per-consumer floor to the shared producer;
        # the td_join combine resolves the fan-out meet.
        td_forward(ctx)
        if ctx.mode == "delta_sum":
            (x,) = ctx.saved_tensors
            return grad_output - x * ((n - 1.0) / n), None, None
        return grad_output / n, None, None


class HardtanhProjection(torch.autograd.Function):
    """
    Projection-aware DyT-style activation: y = clamp(alpha * x, -1, 1).

    Normalization-free alternative to RMSNorm/LayerNorm ("Transformers
    without Normalization"-style dynamic tanh, in its piecewise-linear form).
    Backward is the exact closed-form L2 projection onto the 3-branch graph
    (same template as LeakyReLUProjection).
    """
    @staticmethod
    def forward(ctx, x, alpha=0.5):
        ctx.save_for_backward(x)
        ctx.alpha = float(alpha)
        return torch.clamp(ctx.alpha * x, -1.0, 1.0)

    @staticmethod
    def backward(ctx, z):
        (x,) = ctx.saved_tensors
        result = _hardtanh_impl(x, z, ctx.alpha)
        return process_node_target(x, result, ctx), None



class LeakyReLUProjection(torch.autograd.Function):
    """
    Projection-aware LeakyReLU.

    Forward: y = max(x, negative_slope * x)
    Backward: projects x onto the closest branch-consistent point given z.
    """
    @staticmethod
    def forward(ctx, x, negative_slope=0.01):
        ctx.save_for_backward(x)
        ctx.negative_slope = float(negative_slope)
        return F.leaky_relu(x, negative_slope=ctx.negative_slope)

    @staticmethod
    def backward(ctx, z):
        (x,) = ctx.saved_tensors
        result = _leakyrelu_impl(x, z, ctx.negative_slope)
        return process_node_target(x, result, ctx), None


class LeakyReLULInfinityProjection(torch.autograd.Function):
    """
    Projection-aware LeakyReLU under the L_infinity (Chebyshev) metric.

    Forward: y = max(x, negative_slope * x)
    Backward: projects (x, z) onto the nearest branch of the LeakyReLU graph
    measured by max(|x - x_bar|, |z - f(x_bar)|). Unlike the L2 projection,
    the flat (leaky) branch transports the target at full amplitude instead
    of slope-scaled — see _leakyrelu_linf_impl.
    """
    @staticmethod
    def forward(ctx, x, negative_slope=0.01):
        ctx.save_for_backward(x)
        ctx.negative_slope = float(negative_slope)
        return F.leaky_relu(x, negative_slope=ctx.negative_slope)

    @staticmethod
    def backward(ctx, z):
        (x,) = ctx.saved_tensors
        result = _leakyrelu_linf_impl(x, z, ctx.negative_slope)
        return process_node_target(x, result, ctx), None


class StepProjection(torch.autograd.Function):
    """
    Step activation projection.
    Forward:  step(x) -> +1 if x >= 0 else -1
    Backward: projects x onto the closed half-space agreeing with z_target.
              If step(x) already matches sign(z_target) (z = 0 counts as +1),
              leave x untouched; else move x to the nearest point of the
              other branch: 0 for +1, -finfo.tiny for -1.
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(x >= 0, 1.0, -1.0).to(x.dtype)

    @staticmethod
    def backward(ctx, z_target):
        x, = ctx.saved_tensors
        # sign-agreement mask: keep x where it already produces z_target.
        agrees = (x >= 0) == (z_target >= 0)
        x_proj = torch.where(agrees, x, torch.where(x >= 0, -torch.finfo(x.dtype).tiny, 0.0).to(x.dtype))
        return process_node_target(x, x_proj, ctx)

class QuantizeReLUProjection(torch.autograd.Function):
    """
    Quantized ReLU with Projection-aware gradients.
    
    Forward: 
        f(x) = step * round(max(0, x) / step)
    
    Backward: 
        Projects (x, z_target) onto the nearest segment of the non-negative 
        staircase. The first segment (k=0) extends from (-inf, 0.5 * step].
    """
    @staticmethod
    def forward(ctx, x, step=1.0):
        ctx.save_for_backward(x)
        ctx.step = step
        # Apply ReLU then round to nearest step
        return step * torch.round(torch.clamp(x, min=0.0) / step)

    @staticmethod
    def backward(ctx, z_target):
        x, = ctx.saved_tensors
        s = ctx.step

        # dist(k) is convex in k >= 1; its continuous minimiser over the level t=k*s
        # is t = z + sign(x-z)*max(|x-z| - s/2, 0)/2. Round it (>= 0, ReLU).
        d = x - z_target
        k0 = torch.clamp(torch.round((z_target + d.sign() * (d.abs() - 0.5 * s).clamp_min(0.0) / 2) / s), min=0.0)

        best_x = x.clone()
        best_dist = torch.full_like(x, float('inf'))

        # Check candidate levels: k0-1, k0, k0+1, and the half-line rung k=0
        for k in (k0 - 1, k0, k0 + 1, 0 * k0):
            
            valid_mask = k >= 0
            
            level = k * s
            
            # For k=0, the domain is (-inf, 0.5 * s]
            # For k>0, the domain is [(k - 0.5) * s, (k + 0.5) * s]
            lo = torch.where(k == 0, torch.full_like(k, float('-inf')), (k - 0.5) * s)
            hi = (k + 0.5) * s
            
            x_clamped = torch.maximum(lo, torch.minimum(x, hi))
            dist = (x - x_clamped) ** 2 + (z_target - level) ** 2
            
            better = valid_mask & (dist < best_dist)
            best_x = torch.where(better, x_clamped, best_x)
            best_dist = torch.where(better, dist, best_dist)

        return process_node_target(x, best_x, ctx), None

class GappedStepProjection(torch.autograd.Function):
    """
    Projection-aware Gapped Step activation.

    Like the regular Step activation but with a dead zone of width `delta`
    centred at the origin where the function is undefined:

        f(x) = +1   if x >= delta/2
        f(x) = -1   if x <= -delta/2
        (undefined)  if -delta/2 < x < delta/2

    The projection selects the closer of the two feasible branches:
        Branch 1 (positive): x' = clamp(x, min=delta/2), y = +1
        Branch 2 (negative): x' = clamp(x, max=-delta/2), y = -1
    """
    @staticmethod
    def forward(ctx, x, delta=2.0):
        ctx.save_for_backward(x)
        ctx.delta = delta
        
        # Clamp to the closest non-zero value (+1 or -1). 
        # torch.where ensures we never output exactly 0, even if x == 0.
        return torch.where(x >= 0.0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, z_target):
        x, = ctx.saved_tensors
        half = ctx.delta / 2.0

        # Branch 1: positive (x >= delta/2, y = +1)
        x_pos = torch.clamp(x, min=half)
        dist_pos = (x - x_pos) ** 2 + (z_target - 1.0) ** 2

        # Branch 2: negative (x <= -delta/2, y = -1)
        x_neg = torch.clamp(x, max=-half)
        dist_neg = (x - x_neg) ** 2 + (z_target + 1.0) ** 2

        # Select branch with minimum distance
        result = torch.where(dist_pos <= dist_neg, x_pos, x_neg)

        return process_node_target(x, result, ctx), None


def max_proj_pt_batch(a, z):
    """
    Vectorized projection onto the maximum function graph.
    Processes a batch of arrays simultaneously.
    
    Args:
        a: Tensor of shape (B, P) where P is the patch size (e.g., kernel_h * kernel_w).
        z: Tensor of shape (B, 1) containing the target maximums.
    Returns:
        Projected tensor of shape (B, P).
    """
    P = a.shape[1]
    a_sorted, idx = torch.sort(a, dim=1)
    # Candidate c sets the top P-c sorted entries and the output to their
    # joint mean z_k = (sum_{i>=c} a_i + z) / (P-c+1). It is feasible iff the
    # largest untouched entry a_{c-1} <= z_k; feasibility is monotone in c and
    # the feasible candidate with the fewest moved entries (largest c) is the
    # projection, so no per-candidate distances are needed.
    suffix = a_sorted.flip(1).cumsum(1).flip(1)
    z_k = (suffix + z) / torch.arange(P + 1, 1, -1, device=a.device, dtype=a.dtype)
    pos = torch.arange(P, device=a.device)
    valid = F.pad(a_sorted[:, :-1], (1, 0), value=float('-inf')) <= z_k
    k = (valid * pos).amax(dim=1, keepdim=True)
    best_sorted = torch.where(pos >= k, z_k.gather(1, k), a_sorted)
    return torch.empty_like(a).scatter_(1, idx, best_sorted)


class SeqMaxPoolProjection(torch.autograd.Function):
    """Projection-aware global max-pool over the sequence (token) dimension.

    GNN-style readout: collapses (B, T, D) to (B, D) by taking the max over T.
    Backward: for each independent (b, d) slice, projects the T-element vector
    and the incoming scalar target onto the max-equality constraint via
    `max_proj_pt_batch`.
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.amax(x, dim=1)

    @staticmethod
    def backward(ctx, z_target):
        x, = ctx.saved_tensors
        B, T, D = x.shape

        # Lay out each (b, d) slice as a row of length T
        a_batch = x.permute(0, 2, 1).reshape(B * D, T).contiguous()
        z_batch = z_target.reshape(B * D, 1)

        a_proj_batch = max_proj_pt_batch(a_batch, z_batch)

        x_proj = a_proj_batch.view(B, D, T).permute(0, 2, 1).contiguous()
        return process_node_target(x, x_proj, ctx)


class SeqAvgPoolProjection(torch.autograd.Function):
    """Projection-aware global average-pool over the sequence (token) dimension.

    GNN-style readout: collapses (B, T, D) to (B, D) by taking the mean over T.
    Backward: joint projection of (x, z_target) onto ``mean(x, dim=1) == y``
    (same form as SeqMaxPool): every token moves by ``(z_target - mean(x)) /
    (T + 1)``.
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.mean(dim=1)

    @staticmethod
    def backward(ctx, z_target):
        x, = ctx.saved_tensors
        y = x.mean(dim=1)
        x_proj = x + (z_target - y).unsqueeze(1) / (x.shape[1] + 1)
        return process_node_target(x, x_proj, ctx)


@torch.compile(dynamic=_TC_DYNAMIC, mode=_TC_MODE)
def _maxpool_proj(x, z, kH: int, kW: int):
    """Unpadded max-pool target. Windows never overlap (stride == kernel), so a reshape
    separates them; pixels past the last whole window keep their value. x is NCHW and
    z the pooled target."""
    N, C, H, W = x.shape
    Ho, Wo = z.shape[2:]
    h, w = Ho * kH, Wo * kW
    a = x[:, :, :h, :w].reshape(N, C, Ho, kH, Wo, kW).permute(0, 1, 2, 4, 3, 5).reshape(-1, kH * kW)
    a = max_proj_pt_batch(a, z.reshape(-1, 1))
    a = a.view(N, C, Ho, Wo, kH, kW).permute(0, 1, 2, 4, 3, 5).reshape(N, C, h, w)
    return torch.cat([torch.cat([a, x[:, :, :h, w:]], 3), x[:, :, h:]], 2)


class MaxPool2DProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, kernel_size, stride=None, padding=0):
        if stride is None:
            stride = kernel_size
        
        k_tuple = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        s_tuple = (stride, stride) if isinstance(stride, int) else tuple(stride)
        
        if k_tuple != s_tuple:
            raise ValueError("MaxPool projection requires strides == pool_size")

        ctx.save_for_backward(input)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        
        return F.max_pool2d(input, kernel_size, stride, padding)

    @staticmethod
    def backward(ctx, z_target):
        input, = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        padding = ctx.padding

        if padding == 0 or padding == (0, 0):
            kH, kW = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
            x_bar = _maxpool_proj(input.detach(), z_target.detach(), kH, kW)
            return process_node_target(input, x_bar, ctx), None, None, None

        N, C, H_in, W_in = input.shape
        _, _, H_out, W_out = z_target.shape

        kH = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
        kW = kernel_size[1] if isinstance(kernel_size, tuple) else kernel_size

        # 1. Extract patches using unfold
        # Shape: (N, C * kH * kW, H_out * W_out)
        pH, pW = (padding, padding) if isinstance(padding, int) else padding
        patches = F.unfold(F.pad(input, (pW, pW, pH, pH), value=float('-inf')), kernel_size, stride=stride)
        L = patches.shape[-1] # Number of spatial patches (H_out * W_out)
        
        # 2. Reshape to isolate each local pooling window
        # (N, C, kH * kW, L) -> Permute to (N, C, L, kH * kW)
        patches = patches.view(N, C, kH * kW, L).permute(0, 1, 3, 2).contiguous()
        
        # Flatten batch, channels, and spatial dims into a single batch dimension
        a_batch = patches.view(-1, kH * kW) # Shape: (B, P)
        z_batch = z_target.reshape(-1, 1)      # Shape: (B, 1)

        # 3. Apply vectorized projection
        a_proj_batch = max_proj_pt_batch(a_batch, z_batch)

        # 4. Reconstruct the image
        a_proj_patches = a_proj_batch.view(N, C, L, kH * kW)
        
        a_proj_unfolded = a_proj_patches.permute(0, 1, 3, 2).contiguous().view(N, C * kH * kW, L)

        a_proj = F.fold(
            a_proj_unfolded, 
            output_size=(H_in, W_in), 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding
        )
        covered = F.fold(torch.ones_like(a_proj_unfolded[:1, :kH * kW]), (H_in, W_in), kernel_size,
                         stride=stride, padding=padding) > 0
        return process_node_target(input, torch.where(covered, a_proj, input), ctx), None, None, None


class ConvPatchProjection(torch.autograd.Function):
    """
    Handles the Spatial Consensus for Convolutional Activations.
    """
    @staticmethod
    def forward(ctx, input, kernel_size, stride, padding):
        ctx.save_for_backward(input)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        
        patches = F.unfold(input, kernel_size, dilation=1, padding=padding, stride=stride)
        
        kH = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
        kW = kernel_size[1] if isinstance(kernel_size, tuple) else kernel_size
        sH = stride[0] if isinstance(stride, tuple) else stride
        sW = stride[1] if isinstance(stride, tuple) else stride
        pad_h = padding[0] if isinstance(padding, tuple) else padding
        pad_w = padding[1] if isinstance(padding, tuple) else padding
        
        H_out = (input.shape[2] + 2 * pad_h - kH) // sH + 1
        W_out = (input.shape[3] + 2 * pad_w - kW) // sW + 1
        ctx.H_out = H_out
        ctx.W_out = W_out
        
        return patches.view(input.shape[0], -1, H_out, W_out).permute(0, 2, 3, 1)

    @staticmethod
    def backward(ctx, z_target):
        input, = ctx.saved_tensors
        N, C, H, W = input.shape
        
        kH = ctx.kernel_size[0] if isinstance(ctx.kernel_size, tuple) else ctx.kernel_size
        kW = ctx.kernel_size[1] if isinstance(ctx.kernel_size, tuple) else ctx.kernel_size
        
        L = ctx.H_out * ctx.W_out
        
        z_patches = z_target.permute(0, 3, 1, 2).reshape(N, -1, L)
        
        target_sum = F.fold(
            z_patches, 
            output_size=(H, W), 
            kernel_size=ctx.kernel_size, 
            padding=ctx.padding, 
            stride=ctx.stride
        )
        
        dummy_ones = torch.ones(1, kH * kW, L, device=input.device, dtype=input.dtype)
        overlap_counts = F.fold(
            dummy_ones, 
            output_size=(H, W), 
            kernel_size=ctx.kernel_size, 
            padding=ctx.padding, 
            stride=ctx.stride
        )
        
        target_img = torch.where(overlap_counts > 0, target_sum / overlap_counts, input)
        
        target = process_node_target(input, target_img, ctx)
        
        return target, None, None, None


class Conv2DProjection(torch.autograd.Function):
    """ConvPatchProjection + MatMulProjection for the plain card solve, computed with
    convolutions instead of a patch matrix k^2 times the size of the input (see
    _conv_card_proj). weight is (O, C*kH*kW), bias is (1, O) or None."""
    @staticmethod
    def forward(ctx, input, weight, bias, kernel_size, stride, padding, num_steps, alpha, g):
        ctx.conf = (stride, padding, alpha, g, num_steps)
        w = weight.view(weight.shape[0], input.shape[1], *kernel_size)
        out = F.conv2d(input, w, None if bias is None else bias.squeeze(0), stride, padding)
        ctx.save_for_backward(input, w, bias, out)
        return out

    @staticmethod
    def backward(ctx, z_target):
        x, w, bias, out = ctx.saved_tensors
        stride, padding, alpha, g, num_steps = ctx.conf
        x_bar, w_res, bias_res = _conv_card_proj(
            x.detach(), w.detach(), None if bias is None else bias.detach(), out.detach(), z_target.detach(),
            stride, padding, alpha, g, num_steps, config.matmul_b_self_deflate,
            config.projection_alpha_auto, config.proj_out_cubic, ctx.needs_input_grad[0])
        target = None if x_bar is None else process_node_target(x, x_bar, ctx)
        return (target, _tc_own(w_res.flatten(1)), _tc_own(bias_res),
                None, None, None, None, None, None)


@torch.compile(dynamic=_TC_DYNAMIC, mode=_TC_MODE)
def rmsnorm_proj_exact(x, z, g: float, num_steps: int):
    """Joint L2 projection onto the RMSNorm graph manifold.

    Minimizes ||sigma*u - x||^2 + g*||sqrt(n)*u - z||^2 over sigma >= 0,
    ||u|| = 1, rows on the last dim. The optimal u lies in span{x, z}, so with
    an orthonormal basis (e1 = x/||x||, e2 from Gram-Schmidt on z) the problem
    reduces to maximizing
        J(theta) = <x,u>^2 * 1[<x,u> > 0] + 2*g*sqrt(n)*<z,u>,
        u = cos(theta)*e1 + sin(theta)*e2,
    solved by damped Newton from the legacy init u ∝ z (theta0 = atan2(b2,b1)).
    By construction b2 >= 0, so the optimum has sin(theta) >= 0: theta in
    [0, pi]. Degenerate rows (z ∥ x, or ||z|| ~ 0) collapse gracefully: r = 0
    makes the e2 term vanish, and a vanished target gives theta -> 0, i.e.
    x_bar -> x (no signal, no change — unlike legacy, which renormalizes pure
    noise up to the sqrt(n)-sphere).
    """
    n = x.size(-1)
    sqrt_n = math.sqrt(n)

    a = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(1e-12)
    e1 = x / a
    b1 = (z * e1).sum(dim=-1, keepdim=True)
    r = z - b1 * e1
    b2 = torch.linalg.norm(r, dim=-1, keepdim=True)
    e2 = r / b2.clamp_min(1e-12)

    c = 2.0 * g * sqrt_n
    a2 = a.square()
    theta0 = torch.atan2(b2, b1)

    # Coarse grid init: J can have two local maxima on [0, pi]; Newton from
    # the legacy init alone converges to the wrong one for some rows. Scan a
    # small grid (plus theta0) and refine the best candidate.
    grid = torch.linspace(0.0, math.pi, 33, device=x.device, dtype=x.dtype)
    ct_g = torch.cos(grid)
    st_g = torch.sin(grid)
    j_grid = (a * ct_g).clamp_min(0.0).square() + c * (b1 * ct_g + b2 * st_g)
    theta_grid = grid[j_grid.argmax(dim=-1, keepdim=True)]
    ct0 = torch.cos(theta0)
    st0 = torch.sin(theta0)
    j_theta0 = (a * ct0).clamp_min(0.0).square() + c * (b1 * ct0 + b2 * st0)
    j_best = torch.gather(j_grid, -1, j_grid.argmax(dim=-1, keepdim=True))
    theta = torch.where(j_theta0 >= j_best, theta0, theta_grid)
    theta_init = theta

    for _ in range(num_steps):
        ct = torch.cos(theta)
        st = torch.sin(theta)
        active = (ct > 0).to(x.dtype)
        # J'(theta) and J''(theta); the cos^2 term is C^1 across cos(theta)=0.
        j1 = -a2 * (2.0 * st * ct) * active + c * (b2 * ct - b1 * st)
        j2 = -2.0 * a2 * (ct * ct - st * st) * active - c * (b1 * ct + b2 * st)
        # Newton ascent where curvature is negative; small gradient-ascent
        # step elsewhere. Step clamp keeps the iterate inside [0, pi].
        step = torch.where(j2 < -1e-6,
                           j1 / torch.clamp(j2, max=-1e-6),
                           -0.2 * torch.sign(j1))
        theta = (theta - torch.clamp(step, min=-0.5, max=0.5)).clamp(0.0, math.pi)

    # Safeguard: never end below the init candidate.
    ct = torch.cos(theta)
    st = torch.sin(theta)
    j_final = (a * ct).clamp_min(0.0).square() + c * (b1 * ct + b2 * st)
    ct_i = torch.cos(theta_init)
    st_i = torch.sin(theta_init)
    j_init = (a * ct_i).clamp_min(0.0).square() + c * (b1 * ct_i + b2 * st_i)
    use_final = j_final >= j_init
    ct = torch.where(use_final, ct, ct_i)
    st = torch.where(use_final, st, st_i)

    sigma = (a * ct).clamp_min(0.0)
    return sigma * (ct * e1 + st * e2)


class RMSNormProjection(torch.autograd.Function):
    """
    Projection-aware RMS Normalization.

    Forward:
        z = x / RMS(x)

    Backward modes (config.rmsnorm_backward_mode):
      "legacy": sequential two-step reconstruction —
            z_bar = sqrt(n) * z_target / ||z_target||_2
            sigma_bar = mean(x * z_bar)
            x_bar = sigma_bar * z_bar
        Fully trusts the target direction; sigma_bar < 0 silently inverts the
        token, and a near-zero (vanished) target is renormalized to full
        sqrt(n) magnitude.
      "exact": joint projection onto the graph manifold via
        rmsnorm_proj_exact (weighted by config.rmsnorm_g).
      "detached": treat the per-token scale 1/rms(x) as a constant (detached)
        diagonal map — x_bar = z_target * rms(x). Exactly invertible linear
        transport: preserves the target's direction AND relative magnitude
        (nothing vanishes, nothing is renormalized), at the cost of dropping
        the true Jacobian's x-parallel projection term (td_deflate strips
        activation-parallel components at the next bilinear solve anyway).
      "tangent"/"adjoint": see inline comments below.

    STATUS (2026-07-12, STATUS_LEDGER.md): "adjoint" works only WITH the
    transit-VJP chain (record-stack component) or with merge-equalized
    fan-ins (MERGE_RHO, transport_theory.md T1+T2); standalone off-VJP it is
    FALSIFIED — 3.9768 (+0.35, scale drift: one hop's rms^2 scalar remix vs
    heuristics golfed against tangent, merges left unequalized). "tangent"
    is the off-VJP workhorse (load-bearing in the 3.2465 no-VJP record).
    """
    @staticmethod
    def forward(ctx, x, eps=1e-5):
        ctx.save_for_backward(x)
        ctx.eps = float(eps)
        return F.rms_norm(x, (x.size(-1),), eps=eps)

    @staticmethod
    def backward(ctx, z_target):
        x, = ctx.saved_tensors
        n = x.size(-1)
        sigma_bar = None

        mode = config.rmsnorm_backward_mode
        if mode in ("detached", "tangent", "adjoint") and DIAG is None:
            x_bar = _rmsnorm_transport_impl(
                x, z_target, float(ctx.eps),
                mode != "detached", mode == "adjoint",
            )
        elif config.rmsnorm_backward_mode == "detached":
            rms = x.float().square().mean(dim=-1, keepdim=True).add(ctx.eps).sqrt().to(x.dtype)
            x_bar = z_target * rms
        elif config.rmsnorm_backward_mode == "tangent":
            # Detached transport restricted to the norm's TANGENT space: strip
            # the per-token radial (z-parallel) component of the target
            # displacement before transporting. The true rms_norm VJP projects
            # out exactly this component — the output norm is fixed at sqrt(n),
            # so radial demands ("grow/shrink this token") are unrealizable and
            # backprop annihilates them. Plain "detached" forwards them into
            # the layer below as pure weight/activation inflation (measured:
            # persistent cos(pseudo-grad, W) ~ -0.25 on c_k via q/k head norms
            # -> linear w_max runaway at full size). Tangent mode keeps the
            # invertible-transport property for realizable directions and
            # matches the VJP's null space for the rest.
            rms = x.float().square().mean(dim=-1, keepdim=True).add(ctx.eps).sqrt().to(x.dtype)
            z = x / rms
            delta = z_target - z
            delta = delta - (delta * z).sum(dim=-1, keepdim=True) \
                / z.square().sum(dim=-1, keepdim=True).clamp_min(1e-12) * z
            x_bar = (z + delta) * rms
        elif config.rmsnorm_backward_mode == "adjoint":
            # True-VJP transit metric (trunk-transit probe 2026-07-11): tangent
            # mode transports rms * delta_tangent — displacement units, which
            # AMPLIFY through the norm while true gradients SHRINK by 1/rms.
            # At every residual fan-in the sublayer path crosses a norm the
            # skip path doesn't, so its demand is over-weighted by rms^2
            # (measured 3e4 inversion: block-input demand 99.9% skip+mlp vs
            # true gradient 98% attention-path; combined cos collapses
            # 0.795 -> 0.104). Same tangential projection as "tangent", scaled
            # by 1/rms — the exact rms_norm VJP applied to the displacement.
            rms = x.float().square().mean(dim=-1, keepdim=True).add(ctx.eps).sqrt().to(x.dtype)
            z = x / rms
            delta = z_target - z
            delta = delta - (delta * z).sum(dim=-1, keepdim=True) / n * z
            x_bar = x + delta / rms
        elif config.rmsnorm_backward_mode == "exact":
            x_bar = rmsnorm_proj_exact(
                x, z_target, config.rmsnorm_g, config.rmsnorm_newton_steps
            )
        else:
            # Fused legacy reconstruction:
            # z_bar = sqrt(n) z / ||z||;  x_bar = mean(x*z_bar) * z_bar
            x_bar, sigma_bar = _rmsnorm_legacy_impl(x, z_target)

        if DIAG is not None:
            stats = {
                "rel_delta": float((x_bar - x).norm() / (x.norm() + 1e-12)),
            }
            if sigma_bar is not None:
                stats["sigma_flip"] = float((sigma_bar < 0).float().mean())
            DIAG.record("rmsnorm", **stats)

        return process_node_target(x, x_bar, ctx), None



@torch.compile(dynamic=_TC_DYNAMIC, mode=_TC_MODE)
def _rmsnorm_transport_impl(x, z_target, eps: float, tangential: bool,
                            adjoint: bool):
    """Compiled twin of the detached/tangent/adjoint eager branches in
    RMSNormProjection.backward (kept verbatim below for DIAG runs). tangential
    strips the per-token radial component; adjoint additionally scales by
    1/rms (the exact rms_norm VJP applied to the displacement, whose radial
    removal divides by n, not ||z||^2) instead of rms."""
    rms = x.float().square().mean(dim=-1, keepdim=True).add(eps).sqrt().to(x.dtype)
    if not tangential:
        return z_target * rms
    z = x / rms
    delta = z_target - z
    delta = delta - (delta * z).sum(dim=-1, keepdim=True) \
        / (x.size(-1) if adjoint else z.square().sum(dim=-1, keepdim=True).clamp_min(1e-12)) * z
    if adjoint:
        return x + delta / rms
    return (z + delta) * rms


@torch.compile(dynamic=_TC_DYNAMIC, mode=_TC_MODE)
def _hadamard_linf_b_impl(a, b, z_target):
    # Chebyshev projection of (a, b) onto the hyperbola (a+da)(b+db) = z:
    # at the optimum |da| = |db| = t, so solve the four sign patterns
    # ab*t^2 + (alpha*b + beta*a)*t - r = 0  (alpha, beta in {+-1}, r = z - a*b)
    # and keep the smallest real root t >= 0. Returns b + beta*t for the
    # winning pattern: bounded b-demand (~sqrt(|r|) as a -> 0). a's card target stays the
    # batch-reduced least-squares formula (computed by _hadamard_impl).
    r = z_target - a * b
    big = torch.full_like(r, float("inf"))
    best_t = big
    best_db = torch.zeros_like(r)
    for alpha in (1.0, -1.0):
        for beta in (1.0, -1.0):
            c2 = alpha * beta
            c1 = alpha * b + beta * a
            disc = c1 * c1 + 4.0 * c2 * r
            ok = disc >= 0
            sq = torch.sqrt(disc.clamp_min(0.0))
            for sgn in (1.0, -1.0):
                t = (-c1 + sgn * sq) / (2.0 * c2)
                valid = ok & (t >= 0)
                t = torch.where(valid, t, big)
                take = t < best_t
                best_t = torch.where(take, t, best_t)
                best_db = torch.where(take, beta * t, best_db)
    # r == 0 (already on the hyperbola) leaves best_db = 0; unreachable z
    # (all patterns invalid — cannot happen for finite inputs) keeps b.
    return b + torch.where(torch.isfinite(best_t), best_db,
                           torch.zeros_like(best_db))


class HadamardProjection(torch.autograd.Function):
    """Element-wise bilinear projection for z = a * b.

    a: shape [dim] — learnable Scale weight.
    b: shape [..., dim] — activation (batch/sequence leading dims).

    b_target is the fixed-a L2 projection b + a(z - ab)/(1 + a^2). a is shared
    over batch/seq dims; a_target is the midpoint of a and the least-squares
    fit sum(z*b)/sum(b*b).

    b_target flows on through the projection chain (with td_repeat_target).
    a is a parameter, so a.grad holds the residual a - a_target.
    """
    @staticmethod
    def forward(ctx, a, b):
        z = a * b
        ctx.save_for_backward(a, b, z)
        return z

    @staticmethod
    def backward(ctx, z_target):
        a, b, z = ctx.saved_tensors
        # b_target: fixed-a L2 projection; a_target: inner-product
        # (least-squares) formula — sign-correct and stable, never flips sign
        # for negative b values.
        z_for_a = z_target
        if (config.transit_vjp and config.transit_vjp_card_cap
                and config.td_eps_lin > 0.0):
            # card-side consumption window for the Scale card (same global
            # per-tensor scalar as the matmul card cap): the raw conserved
            # trunk demand is huge in displacement units and churns a via the
            # LS formula (scale_min dove to 0.019 with only matmuls capped).
            dz = z_target - z
            lim = config.td_eps_lin * z.detach().norm()
            z_for_a = z + dz * (lim / dz.norm().clamp_min(1e-30)).clamp(max=1.0)
        a_res, b_bar = _hadamard_impl(a, b, z_for_a, 1e-8)
        if config.transit_vjp:
            # two-channel split: a keeps its LS card target; the transit to b
            # is the exact VJP of the displacement (delta_b = delta_z * a).
            b_bar = b + (z_target - z) * a
        elif config.hadamard_norm == "linf":
            # Chebyshev b-transport (a keeps the LS card target above).
            b_bar = _hadamard_linf_b_impl(a, b, z_target)
        return _tc_own(a_res), td_repeat_target(b, b_bar, ctx)  # a_res: the Scale weight's residual


class Conversion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input
    @staticmethod
    def backward(ctx, z_target):
        (input,) = ctx.saved_tensors
        # Graph scope: this is the embedding bridge — a terminal node. Consume
        # (drop) any floor routed here so it never reaches the embedding.
        td_get(ctx)
        # Convert projection target into gradient: push input toward target
        grad = (input - z_target)
        return grad


class ParameterOperand(Conversion):
    """Forward: a copy of input, which is a Parameter, a view of one, or an op's output
    on one. Backward: the target arriving at it becomes the residual input - target
    that Parameters hold in .grad, and activation, the same value computed through
    the op's activation operands, receives the target unchanged.

    Its output has no _base. With no activation it still counts as a Parameter operand,
    so that a later op can select part of it: that op converts in its turn and
    defer_conversion makes this node pass the residual on, which keeps it to one
    conversion per path and keeps the zeros of an unselected coordinate at 0."""
    @staticmethod
    def forward(ctx, input, activation):
        ctx.save_for_backward(input)
        ctx.deferred = False
        return input.clone()

    @staticmethod
    def backward(ctx, z_target):
        if ctx.deferred:
            return z_target, None
        return Conversion.backward(ctx, z_target), z_target if ctx.needs_input_grad[1] else None


def _copy_node(base):  # the ParameterOperand node of a Parameter's copy, if base is one
    node = base.grad_fn
    return node if type(node) is ParameterOperand._backward_cls and not node.needs_input_grad[1] else None


def is_parameter_operand(x):
    if not (isinstance(x, torch.Tensor) and x.requires_grad):
        return False
    base = x._base if x._is_view() else x
    # issubclass: isinstance goes through Parameter's Python metaclass, 3x the cost here.
    return issubclass(type(base), torch.nn.Parameter) or _copy_node(base) is not None


def defer_conversion(x):
    """x goes into a new ParameterOperand: a copy it comes from no longer converts."""
    node = _copy_node(x._base if x._is_view() else x)
    if node is not None:
        node.deferred = True


def parameter_operand(x):
    """Route a Parameter, a view of one or a copy of one into a node that passes
    targets; anything else is returned as it is. Internal: overrides.py applies it,
    and ProjectionModule to its inputs, and overrides.py lists what is covered."""
    if not is_parameter_operand(x):
        return x
    defer_conversion(x)
    return ParameterOperand.apply(x, None)
