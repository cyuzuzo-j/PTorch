import torch
import torch.nn.functional as F
import warnings
from ..config import config
import math
from itertools import repeat

# Diagnostics sink. Scripts (e.g. experiments/attention/diag_targets.py) may
# set `ops.DIAG` to an object exposing `record(key: str, **scalars)`; the
# uncompiled autograd backward wrappers then emit target statistics. Must stay
# None in production — every instrumentation site guards on `DIAG is not None`
# so the default path is unchanged. Never read this inside @torch.compile
# bodies.
DIAG = None

# TD(λ) eligibility trace over depth (config.td_lambda). Seeded with the
# loss-node residual in the loss projection's backward (which autograd runs
# first), then updated by each MatMulProjection.backward as the engine walks
# the chain deep→shallow. Module-level state is sound only for sequential
# chain graphs executed by a single backward at a time. "seed_norm" backs the
# optional td_clip safeguard; "floor" is the scalar trace of norm mode
# (Variant B).
TD_TRACE = {"g": None, "seed_norm": None, "floor": None}


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
def td_repeat_target(x_det, x_bar):
    """Rescale a non-bilinear node's input-target residual up to the current
    trace floor (capped into the solver's linear-transport window), without
    updating the floor — λ-blending happens only at the bilinear solves, so
    activation/pooling/norm nodes act as amplitude repeaters between them.
    Exact no-op at td_lambda = 0 or outside norm mode."""
    if config.td_lambda == 0.0 or config.td_mode != "norm":
        return x_bar
    floor = TD_TRACE.get("floor")
    if floor is None:
        return x_bar
    r = x_det - x_bar
    r_norm = float(r.norm())
    amp = floor
    if config.td_eps_lin > 0.0:
        amp = min(amp, config.td_eps_lin * float(x_det.norm()))
    scale = max(1.0, amp / (r_norm + 1e-12))
    if scale == 1.0:
        return x_bar
    return x_det - scale * r


def process_node_target(x_det, x_bar):
    """process_activation_target followed by the TD repeater; used by every
    non-bilinear projection backward (the bilinear matmul solves keep the raw
    process_activation_target and run the full floor update instead)."""
    out = process_activation_target(x_det, x_bar)
    return td_repeat_target(x_det, out)
# ── end TD repeater ───────────────────────────────────────────────────────────


@torch.compile(dynamic=True)
def matmul_proj_linf(A, B, Z, eps_init=None, g=1.0, omega=1.0, num_steps=5):
    """
    Exact independent bilinear projection for A @ B = Z using the L_infinity (Chebyshev) norm.    
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

    # Caching logic for eps (the L_inf radius)
    if eps_init is not None and eps_init.shape == P.shape:
        eps = eps_init.to(P.device)
    else:
        eps = torch.zeros_like(P)

    # 2. Precompute coordinate rotations (M, 1, K) and (1, N, K) 
    # Expanded implicitly to (M, N, K) during fused operations
    A_ext = A.unsqueeze(-1)       # (..., M, K, 1)
    B_ext = B.unsqueeze(-3)   # (..., 1, K, N)
    S_ext = S.unsqueeze(-2)       # (..., M, 1, N)
    
    U_sum = A_ext + B_ext
    V_diff = A_ext - B_ext
    
    u_abs = U_sum.abs() * 0.5
    v_abs = V_diff.abs() * 0.5
    
    S_plus = torch.sign(U_sum)
    S_minus = torch.sign(V_diff)
    
    # Align u and v based on the geometric case (S)
    # S == 1  -> term_A tracks u_abs, term_B tracks v_abs
    # S == -1 -> term_A tracks v_abs, term_B tracks u_abs
    is_case_A = (S_ext == 1)
    
    base_A = torch.where(is_case_A, u_abs, v_abs)
    base_B = torch.where(is_case_A, v_abs, u_abs)
    
    target_penalty = omega / g
    damping = 1e-5
    
    # 3. Newton's Method to find the optimal L_inf boundary (eps)
    for _ in range(num_steps):
        eps_ext = eps.unsqueeze(-2) # (..., M, 1, N)
        
        # Evaluate piecewise branches for the expanding L1 diamond
        term_A = 2.0 * eps_ext * base_A + eps_ext.square()
        term_B = 2.0 * eps_ext * base_B - eps_ext.square()
        
        # Determine which vertices of the diamond have crossed the axis
        wins_A = term_A >= term_B # (..., M, K, N)
        
        # Reconstruct the objective function and derivatives piecewise
        max_terms = torch.where(wins_A, term_A, term_B)
        f_val = max_terms.sum(dim=-2) + eps * target_penalty - D_abs
        
        deriv_A = 2.0 * base_A + 2.0 * eps_ext
        deriv_B = 2.0 * base_B - 2.0 * eps_ext
        
        f_prime_k = torch.where(wins_A, deriv_A, deriv_B)
        f_prime_val = f_prime_k.sum(dim=-2) + target_penalty
        
        # Damped Newton step
        step = f_val / (f_prime_val + damping)
        
        # L_inf radius must remain strictly positive
        eps = torch.clamp(eps - step, min=0.0)

    # 4. Analytical Consensus Reconstruction
    eps_ext = eps.unsqueeze(-2)
    
    # Re-evaluate winning branches with the highly accurate, final eps
    term_A = 2.0 * eps_ext * base_A + eps_ext.square()
    term_B = 2.0 * eps_ext * base_B - eps_ext.square()
    wins_A = term_A >= term_B
    
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
    B_proj = B + dB.mean(dim=-3)
    
    Z_proj = Z - S * (eps / (g * omega))
            
    return A_proj, B_proj, Z_proj, eps.detach()


class MatMulProjectionLinf(torch.autograd.Function):
    """
    Minimize || A_{new} - A_{old} ||_inf + || B_{new} - B_{old} ||_inf + g * || Z_{new} - Z_{old} ||_inf
    subject to A_{new} @ B_{new} = Z_{new} (or A_{new} - A_{new} @ B_{new} = Z_{new} if residual)
    """
    @staticmethod
    def forward(ctx, A, B, num_steps, g, omega, proj_cache=None):
        ctx.save_for_backward(A, B)
        ctx.g = g
        ctx.num_steps = num_steps
        ctx.proj_cache = proj_cache  
        ctx.omega = omega
        return (A @ B) / omega

    @staticmethod
    def backward(ctx, Z_target):
        A, B = ctx.saved_tensors
        A_det = A.detach()
        B_det = B.detach()
        Z_det = Z_target.detach()

        eps_init = None
        if ctx.proj_cache is not None:
            eps_init = ctx.proj_cache.get('eps')

        if (A_det.ndim > 2 or B_det.ndim > 2):
            A_2d = A_det.reshape(-1, A_det.shape[-1]).clone()
            Z_2d = Z_det.reshape(-1, Z_det.shape[-1]).clone() * ctx.omega
            B_2d = B_det.reshape(-1, B_det.shape[-2], B_det.shape[-1]).mean(dim=0).clone()
            
            eps_init_2d = eps_init.reshape(Z_2d.shape) if (eps_init is not None and eps_init.shape == Z_det.shape) else None
            
            A_proj_2d, B_proj_2d, _, eps_new = matmul_proj_linf(
                A_2d, B_2d, Z_2d, eps_init=eps_init_2d, g=ctx.g, 
                omega=ctx.omega, num_steps=ctx.num_steps
            )
            
            A_proj = A_proj_2d.reshape(A_det.shape)
            B_proj = B_proj_2d.reshape(B_det.shape[-2], B_det.shape[-1]).expand(B_det.shape)
            
            if ctx.proj_cache is not None:
                ctx.proj_cache['eps'] = eps_new.reshape(Z_det.shape)
        else:
            A_proj, B_proj, _, eps_new = matmul_proj_linf(
                A_det.contiguous().clone(), B_det.contiguous().clone(), Z_det.contiguous().clone() * ctx.omega,
                eps_init=eps_init, g=ctx.g, omega=ctx.omega, 
                num_steps=ctx.num_steps
            )

            if ctx.proj_cache is not None:
                ctx.proj_cache['eps'] = eps_new

        return process_activation_target(A_det, A_proj), B_proj, None, None, None, None, None, None        

@torch.compile(dynamic=True)
def matmul_proj(A, B, Z, t_init=None, alpha=1.0, g=1.0, omega=1.0, num_steps=1, residual=False):
    """
    Exact bilinear projection for A @ B = Z (or A - A @ B = Z if residual=True).
    """
    M = A.size(-2)
    N = B.size(-1)


    # 1. Compute pairwise operations in O(M*N) without 3D expansion
    p = A @ B # (..., M, N)
    qa = (A * A).sum(dim=-1, keepdim=True)  # (..., M, 1)
    qb = (B * B).sum(dim=-2, keepdim=True)  # (..., 1, N)
    q_eff = qa + alpha * qb  # (..., M, N)

    # 2. Caching logic for t
    if t_init is not None and t_init.shape == p.shape:
        t = t_init.to(p.device)
    else:
        t = torch.zeros_like(p)

    target_penalty = (omega ** 2) / (g ** 2)

    # --- NUMERICAL SAFEGUARDS ---
    eps = 1e-5             # Minimum distance from the singularity
    max_step_size = 0.5    # Maximum allowable change in t per step

    # Calculate strict boundary for t to prevent alpha - t^2 from approaching 0
    # We require alpha - t^2 >= eps  =>  t^2 <= alpha - eps
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
    lo = torch.full_like(p, -max_t_val)   # f(lo) < 0
    hi = torch.full_like(p, max_t_val)    # f(hi) > 0

    for _ in range(num_steps):
        t2 = t.square()
        alpha_minus_t2 = alpha - t2 # Guaranteed to be >= eps

        N_num = alpha * (p * (alpha + t2) + t * q_eff)
        f_val = (N_num / (alpha_minus_t2.square())) - Z + t * target_penalty

        N_prime = alpha * (2.0 * t * p + q_eff)
        f_prime_val = ((N_prime * alpha_minus_t2) + 4.0 * t * N_num) / (alpha_minus_t2 ** 3) + target_penalty

        # f is strictly increasing: f>0 => root lies left of t (tighten hi), else lo.
        lo = torch.where(f_val < 0, t, lo)
        hi = torch.where(f_val > 0, t, hi)

        # Movement-limited Newton step (f_prime_val > 0 by monotonicity).
        step = torch.clamp(f_val / f_prime_val.clamp_min(1e-12), min=-max_step_size, max=max_step_size)
        t_newton = t - step

        # Accept Newton only if it stays strictly inside the bracket; else bisect.
        # NaN/inf steps fail the comparison and fall back to bisection automatically.
        inside = (t_newton > lo) & (t_newton < hi)
        t_next = torch.where(inside, t_newton, 0.5 * (lo + hi))

        # Freeze elements that have already hit the root (keeps idempotence exact).
        converged = f_val.abs() < 1e-10
        t = torch.where(converged, t, t_next)

    # 4. Analytical Consensus Reconstruction
    t2 = t.square()
    denom = alpha - t2
    inv_denom = 1.0 / denom       # (M, N) - Safe because denom >= eps
    t_inv_denom = t / denom       # (M, N)

    # A_proj analytically averages over N proposals via Matrix Math
    sum_inv_denom_j = inv_denom.sum(dim=-1, keepdim=True) # (..., M, 1)
    A_proj = (alpha / N) * (A * sum_inv_denom_j + t_inv_denom @ B.transpose(-2, -1))

    # B_proj analytically averages over M proposals via Matrix Math
    sum_inv_denom_i = inv_denom.sum(dim=-2, keepdim=True) # (..., 1, N)
    B_proj = (1.0 / M) * (alpha * B * sum_inv_denom_i + A.transpose(-2, -1) @ t_inv_denom)

    Z_proj = Z - t * target_penalty
    return A_proj, B_proj, Z_proj, t.detach()


# ─── autograd.Function wrappers ───────────────────────────────────────────────
class MatMulProjection(torch.autograd.Function):
    """
    Minimize || A_{new} - A_{old} ||_F^2 + alpha * || B_{new} - B_{old} ||_F^2 + g * || Z_{new} - Z_{old} ||_F^2
    subject to A_{new} @ B_{new} = Z_{new} (or A_{new} - A_{new} @ B_{new} = Z_{new} if residual)
    """
    @staticmethod
    def forward(ctx, A, B, num_steps, alpha, g, omega, proj_cache=None, pairwise=False, forward_cache=None):
        ctx.alpha = alpha
        ctx.g = g
        ctx.num_steps = num_steps
        ctx.proj_cache = proj_cache
        ctx.forward_cache = forward_cache
        ctx.omega = omega
        ctx.pairwise = pairwise
        ctx.save_for_backward(A, B)
        return (A @ B) / omega

    @staticmethod
    def backward(ctx, Z_target):
        A, B = ctx.saved_tensors
        A_det = A.detach()
        B_det = B.detach()
        Z_det = Z_target.detach()

        alpha = ctx.alpha
        if config.projection_alpha_auto:
            # Rescale so moving A and moving B cost comparably; one scalar
            # sync per layer per backward. Same exact projection, computed in
            # a per-layer-balanced metric.
            qa_mean = A_det.square().sum(dim=-1).mean()
            qb_mean = B_det.square().sum(dim=-2).mean()
            ratio = float((qa_mean / qb_mean.clamp_min(1e-12)).clamp(1e-3, 1e6))
            # Quantize to powers of two: alpha is a python-scalar argument of
            # the compiled matmul_proj, so a fresh value every step would
            # trigger unbounded torch.compile specializations.
            alpha = alpha * (2.0 ** round(math.log2(ratio)))

        t_init = None
        if not ctx.pairwise and ctx.proj_cache is not None:
            t_init = ctx.proj_cache.get('t')

        if not ctx.pairwise and (A_det.ndim > 2 or B_det.ndim > 2):
            A_2d = A_det.reshape(-1, A_det.shape[-1]).clone()
            Z_2d = Z_det.reshape(-1, Z_det.shape[-1]).clone() * ctx.omega
            B_2d = B_det.reshape(-1, B_det.shape[-2], B_det.shape[-1]).mean(dim=0).clone()
            
            t_init_2d = t_init.reshape(Z_2d.shape) if (t_init is not None and t_init.shape == Z_det.shape) else None
            
            A_proj_2d, B_proj_2d, Z_proj_2d, t_new = matmul_proj(
                A_2d, B_2d, Z_2d, t_init=t_init_2d, alpha=alpha, g=ctx.g,
                omega=ctx.omega, num_steps=ctx.num_steps
            )
            
            A_proj = A_proj_2d.reshape(A_det.shape)
            B_proj = B_proj_2d.reshape(B_det.shape[-2], B_det.shape[-1]).expand(B_det.shape)
            Z_proj = Z_proj_2d.reshape(Z_det.shape)

            if ctx.proj_cache is not None:
                ctx.proj_cache['t'] = t_new.reshape(Z_det.shape)
        else:
            # .contiguous().clone() ensures no view _base for 2D inputs
            # that may come from transpose() or other view ops
            A_proj, B_proj, Z_proj, t_new = matmul_proj(
                A_det.contiguous().clone(), B_det.contiguous().clone(), Z_det.contiguous().clone() * ctx.omega,
                t_init=t_init, alpha=alpha, g=ctx.g, omega=ctx.omega,
                num_steps=ctx.num_steps
            )

            # Update the cache for the next iteration if not pairwise
            if not ctx.pairwise and ctx.proj_cache is not None:
                ctx.proj_cache['t'] = t_new

            B_proj = B_proj

        if ctx.forward_cache is not None:
            ctx.forward_cache[0] = Z_proj

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

        A_ret = process_activation_target(A_det, A_proj)

        if config.td_lambda != 0.0 and TD_TRACE.get("g") is not None:
            lam = config.td_lambda
            if config.td_mode == "norm":
                # Variant B: scalar floor trace — amplify the local (correctly
                # Bᵀ-transported) direction up to a seed-coupled magnitude
                # floor. Only a norm crosses layers, so this is frame-invariant
                # and width-agnostic. The bias pad column is included in the
                # rescale (its target is discarded by F.pad's backward anyway;
                # its contribution to ‖r_local‖ is negligible).
                floor = TD_TRACE["floor"]
                r_local = A_det - A_ret
                if config.td_deflate:
                    # Strip the per-row activation-parallel artifact; keep only
                    # the transported direction (see config.td_deflate).
                    coef = (r_local * A_det).sum(-1, keepdim=True) \
                        / A_det.square().sum(-1, keepdim=True).clamp_min(1e-12)
                    r_local = r_local - coef * A_det
                r_norm = float(r_local.norm())
                floor = lam * floor + (1.0 - lam) * r_norm
                if config.td_clip > 0.0 and TD_TRACE["seed_norm"] is not None:
                    floor = min(floor, config.td_clip * float(TD_TRACE["seed_norm"]))
                TD_TRACE["floor"] = floor
                # Write amplitude: the seed-coupled floor, capped into the
                # solver's linear-transport window (see config.td_eps_lin).
                amp = floor
                if config.td_eps_lin > 0.0:
                    amp = min(amp, config.td_eps_lin * float(A_det.norm()))
                scale = max(1.0, amp / (r_norm + 1e-12))
                A_ret = A_det - scale * r_local
                if DIAG is not None:
                    DIAG.record(
                        "td_trace",
                        r_local_norm=r_norm,
                        g_norm=scale * r_norm,
                        scale=scale,
                    )
            else:
                # Variant A: vector trace — blend the residual vector itself.
                g = TD_TRACE["g"]
                # Bias is folded in as a padded ones column, so A may be one
                # column wider than the trace; the pad column's target is
                # discarded by F.pad's backward, so only the first d columns
                # carry signal.
                d = g.shape[-1]
                r_local = A_det[..., :d] - A_ret[..., :d]
                if g.shape == r_local.shape:
                    g_new = (lam * g + (1.0 - lam) * r_local).detach()
                    if config.td_clip > 0.0 and TD_TRACE["seed_norm"] is not None:
                        cap = config.td_clip * TD_TRACE["seed_norm"]
                        g_new = g_new * (cap / g_new.norm().clamp_min(1e-12)).clamp(max=1.0)
                    TD_TRACE["g"] = g_new
                    A_ret = torch.cat([A_det[..., :d] - g_new, A_ret[..., d:]], dim=-1)
                # Width mismatch: leave the target purely local (no dimension bridge).
                if DIAG is not None:
                    DIAG.record(
                        "td_trace",
                        r_local_norm=float(r_local.norm()),
                        g_norm=float(TD_TRACE["g"].norm()),
                    )

        return A_ret, B_proj, None, None, None, None, None, None, None


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
            seed = (predictions - out).detach()
            TD_TRACE["g"] = seed
            TD_TRACE["seed_norm"] = seed.norm()
            TD_TRACE["floor"] = float(TD_TRACE["seed_norm"])
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
        
        # Calculate standard cross entropy loss to return as the scalar loss
        if labels.ndim == logits.ndim - 1:
            loss = F.cross_entropy(logits, labels)
        else:
            loss = F.cross_entropy(logits, labels.argmax(dim=-1))
            
        return loss

    @staticmethod
    def backward(ctx, z_target):
        logits, labels = ctx.saved_tensors
        lmbda = ctx.lmbda
        steps = ctx.num_steps

        x = logits
        target_probs = labels
        if labels.ndim == logits.ndim - 1:
            target_probs = F.one_hot(labels, num_classes=logits.size(-1)).to(logits.dtype)

        for _ in range(steps):
            x = x + lmbda * (target_probs - F.softmax(x, dim=-1))

        if config.td_lambda != 0.0:
            seed = (logits - x).detach()
            TD_TRACE["g"] = seed
            TD_TRACE["seed_norm"] = seed.norm()
            TD_TRACE["floor"] = float(TD_TRACE["seed_norm"])

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
    logging. Backward ignores the incoming grad and returns *targets* (one
    gradient-ascent step on  w * logp(a | mu, log_std), iterated num_steps
    times), which ProjectionAdam turns into pseudo-grads (p - p_target):

        d logp / d mu       = (a - mu) / sigma^2
        d logp / d log_std  = ((a - mu) / sigma)^2 - 1

        mu       <- mu       + lambda * w * (a - mu) / sigma^2
        log_std  <- log_std  + lambda * mean_B[ w * (((a-mu)/sigma)^2 - 1) ]

    mu_target is per-sample (B, A) and flows back through the projection layers
    like the discrete logits target. log_std is a single shared (A,) parameter,
    so its per-sample updates are reduced over the batch (mirroring the .mean()
    in the forward loss) and written straight into log_std.grad. Its update is
    unbounded (unlike onehot - softmax), so log_std is clamped to keep sigma
    sane.

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
        mu_t = mu.clone()
        ls = log_std.clone()
        for _ in range(ctx.num_steps):
            std = torch.exp(ls)
            delta = ctx.lmbda * w * (act - mu_t) / std ** 2
            if ctx.max_delta is not None:
                delta = delta.clamp(-ctx.max_delta, ctx.max_delta)
            mu_t = mu_t + delta
            ls = ls + ctx.lmbda * (w * (((act - mu_t) / std) ** 2 - 1.0)).mean(0)
            ls = ls.clamp(GaussianPolicyGradientProjection.LOG_STD_MIN,
                          GaussianPolicyGradientProjection.LOG_STD_MAX)
        # mu target, log_std target; trailing Nones match the extra fwd args
        return mu_t, ls, None, None, None, None, None

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
        cond_0 = (labels <= 0) & (logits > 0)
        cond_1 = (labels > 0) & (logits < labels)

        new_logits = logits.clone()
        new_logits[cond_0] = 0.0
        new_logits[cond_1] = labels[cond_1]

        return new_logits, None 


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


class Conversion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input

    @staticmethod
    def backward(ctx, z_target):
        (input,) = ctx.saved_tensors
        # Convert projection target into gradient: push input toward target
        grad = (input - z_target)
        return grad

class ReLULInfinityProjection(torch.autograd.Function):
    """
    Sets the new input directly based on the target z,
    projecting onto the ReLU graph using the L-infinity norm.
    """
    @staticmethod
    def forward(ctx, x, forward_cache=None):
        ctx.save_for_backward(x)
        ctx.forward_cache = forward_cache

        return torch.relu(x)

    @staticmethod
    def backward(ctx, z):
        x, = ctx.saved_tensors

        # Solution 1: project onto inactive branch (x <= 0, output = 0)
        x_1 = torch.clamp(x, max=0)
        # L-infinity distance: max(|x - x_1|, |z - 0|)
        dist_1 = torch.max(torch.abs(x - x_1), torch.abs(z))

        # Solution 2: project onto active branch (x > 0, output = x)
        x_2 = torch.clamp((x + z) / 2.0, min=0)
        # L-infinity distance: max(|x - x_2|, |z - x_2|)
        dist_2 = torch.max(torch.abs(x - x_2), torch.abs(z - x_2))

        # Select solution minimizing L-infinity distance
        result = torch.where(dist_1 < dist_2, x_1, x_2)
        
        result_forwards = torch.where(dist_1 < dist_2, torch.zeros_like(x), x_2)
        ctx.forward_cache[0] = result_forwards        

        return process_node_target(x, result), None
    
class ReLUProjection(torch.autograd.Function):
    """
    Sets the new input directly based on the target z.
    Ignores the original input x entirely.
    """
    @staticmethod
    def forward(ctx, x, forward_cache=None):
        ctx.save_for_backward(x)
        ctx.forward_cache = forward_cache
        return torch.relu(x)

    @staticmethod
    def backward(ctx, z):
        x, = ctx.saved_tensors

        # Solution 1: project onto inactive branch (x <= 0, output = 0)
        x_1 = torch.clamp(x, max=0)
        dist_1 = (x - x_1)**2 + z**2

        # Solution 2: project onto active branch (x > 0, output = x)
        x_2 = torch.clamp((x + z) / 2.0, min=0)
        dist_2 = (x - x_2)**2 + (z - x_2)**2

        # Select solution minimizing distance
        result_backwards = torch.where(dist_1 < dist_2, x_1, x_2)
        result_forwards = torch.where(dist_1 < dist_2, torch.zeros_like(x), x_2)
        ctx.forward_cache[0] = result_forwards        
        return process_node_target(x, result_backwards), None    


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
    """
    @staticmethod
    def forward(ctx, x, forward_cache=None):
        ctx.save_for_backward(x)
        ctx.forward_cache = forward_cache
        return F.softmax(x, dim=-1)

    @staticmethod
    def backward(ctx, z):
        x, = ctx.saved_tensors
        eps = 1e-8
        mode = config.softmax_target_mode

        # Handle -inf in x (e.g., from causal masking)
        mask = (x != float('-inf'))
        safe_x = torch.where(mask, x, torch.zeros_like(x))

        if mode == "legacy":
            # Step 1: positivity + KL projection onto the simplex (L1 normalization).
            z_pos = torch.clamp(z, min=eps)
            z_bar = z_pos / z_pos.sum(dim=-1, keepdim=True)

            # Step 2: shift log(z_bar) to be closest in L2 to x along the
            # softmax-equivalence ray.
            log_z = torch.log(z_bar)
            diff = (safe_x - log_z) * mask
            c = diff.sum(dim=-1, keepdim=True) / mask.sum(dim=-1, keepdim=True).clamp(min=1)
            x_bar = log_z + c
            x_bar = torch.where(mask, x_bar, x)
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
            x_bar = torch.where(mask, x_bar, x)

        if ctx.forward_cache is not None:
            ctx.forward_cache[0] = z_bar

        if DIAG is not None:
            disp = torch.where(mask, x_bar - safe_x, torch.zeros_like(safe_x)).abs()
            DIAG.record(
                "softmax",
                clamp_frac=float(((z < eps) & mask).float().mean()),
                z_min=float(z.min()),
                max_logit_disp=float(disp.max()),
                mean_logit_disp=float(disp.mean()),
            )

        return process_node_target(x, x_bar), None

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
        # If mask is -inf, replace the incoming target with x
        target_x = torch.where(mask == float('-inf'), x, target_out)
        return process_node_target(x, target_x), None

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
    """
    @staticmethod
    def forward(ctx, x, num_branches, mode="mean"):
        ctx.num_branches = float(num_branches)
        ctx.mode = mode
        if mode == "delta_sum":
            ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        n = ctx.num_branches
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
        a = ctx.alpha
        inv_a = 1.0 / a

        # Branch low: x <= -1/alpha, y = -1
        x_1 = torch.clamp(x, max=-inv_a)
        dist_1 = (x - x_1) ** 2 + (z + 1.0) ** 2

        # Branch mid: y = alpha * x on [-1/alpha, 1/alpha]
        x_2 = torch.clamp((x + a * z) / (1.0 + a * a), min=-inv_a, max=inv_a)
        dist_2 = (x - x_2) ** 2 + (z - a * x_2) ** 2

        # Branch high: x >= 1/alpha, y = 1
        x_3 = torch.clamp(x, min=inv_a)
        dist_3 = (x - x_3) ** 2 + (z - 1.0) ** 2

        result = torch.where(dist_2 <= dist_1, x_2, x_1)
        best = torch.minimum(dist_1, dist_2)
        result = torch.where(dist_3 < best, x_3, result)

        return process_node_target(x, result), None



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
        slope = ctx.negative_slope

        # Branch 1 (inactive): y = slope * x, x <= 0
        x_1_unconstrained = (x + slope * z) / (1.0 + slope * slope)
        x_1 = torch.clamp(x_1_unconstrained, max=0)
        y_1 = slope * x_1
        dist_1 = (x - x_1) ** 2 + (z - y_1) ** 2

        # Branch 2 (active): y = x, x >= 0
        x_2_unconstrained = (x + z) / 2.0
        x_2 = torch.clamp(x_2_unconstrained, min=0)
        y_2 = x_2
        dist_2 = (x - x_2) ** 2 + (z - y_2) ** 2

        result = torch.where(dist_1 < dist_2, x_1, x_2)
        return process_node_target(x, result), None


class StepProjection(torch.autograd.Function):
    """
    Step activation projection.
    Forward:  step(x) -> +1 if x >= 0 else -1
    Backward: projects x onto the closed half-space agreeing with z_target.
              If sign(x) already matches z_target, leave x untouched; else
              push x to 0 (the boundary closest to x in the wrong branch).
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(x >= 0,
                           torch.tensor(1.0, dtype=x.dtype, device=x.device),
                           torch.tensor(-1.0, dtype=x.dtype, device=x.device))

    @staticmethod
    def backward(ctx, z_target):
        x, = ctx.saved_tensors
        # sign-agreement mask: keep x where it already produces z_target.
        agrees = (x * z_target) >= 0
        x_proj = torch.where(agrees, x, torch.zeros_like(x))
        return process_node_target(x, x_proj)

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

        # Nearest rung index to target, but forced to be at least 0 (ReLU)
        k0 = torch.clamp(torch.round(z_target / s), min=0.0)

        best_x = x.clone()
        best_dist = torch.full_like(x, float('inf'))

        # Check candidate levels: k0-1, k0, k0+1
        for dk in (-1, 0, 1):
            k = k0 + dk
            
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

        return process_node_target(x, best_x), None

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

        return process_node_target(x, result), None


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
    B, P = a.shape
    
    # 1. Sort arrays
    a_sorted, idx = torch.sort(a, dim=1)

    # 2. Compute candidate maxima (z_k)
    a_sorted_flipped = torch.flip(a_sorted, dims=[1])
    cumsum_flipped = torch.cumsum(a_sorted_flipped, dim=1)
    divisors = torch.arange(2, P + 2, device=a.device, dtype=a.dtype).unsqueeze(0)
    
    z_k_flipped = (cumsum_flipped + z) / divisors
    z_k = torch.flip(z_k_flipped, dims=[1])

    # 3. Compute candidate arrays (a_k)
    # Create upper triangular mask (1, P, P)
    i_ge_k = torch.triu(torch.ones((P, P), dtype=torch.bool, device=a.device)).unsqueeze(0)
    
    # a_k shape: (B, P_candidate, P_element)
    a_k = torch.where(i_ge_k, z_k.unsqueeze(2), a_sorted.unsqueeze(1))

    # 4. Compute distances
    dist = ((a_k - a_sorted.unsqueeze(1)) ** 2).sum(dim=2) + (z_k - z) ** 2

    # 5. Select valid candidates
    # Add a small epsilon (1e-5) to handle floating point inaccuracies
    valid = torch.max(a_k, dim=2)[0] <= z_k + 1e-5
    dist_valid = torch.where(valid, dist, torch.tensor(float('inf'), device=a.device, dtype=dist.dtype))

    # 6. Select candidate minimizing distance
    k = torch.argmin(dist_valid, dim=1)

    # Extract the best sorted candidate array for each batch item
    batch_indices = torch.arange(B, device=a.device)
    best_a_k_sorted = a_k[batch_indices, k, :]

    # 7. Unsort back to original spatial layout
    a_proj = torch.zeros_like(a)
    a_proj.scatter_(1, idx, best_a_k_sorted)

    return a_proj


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
        return process_node_target(x, x_proj)


class SeqAvgPoolProjection(torch.autograd.Function):
    """Projection-aware global average-pool over the sequence (token) dimension.

    GNN-style readout: collapses (B, T, D) to (B, D) by taking the mean over T.
    Backward: for the linear constraint ``mean(x_proj, dim=1) == z_target``, the
    minimum-norm correction adds the per-(b, d) residual ``z_target - mean(x)``
    uniformly across the T tokens.
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.mean(dim=1)

    @staticmethod
    def backward(ctx, z_target):
        x, = ctx.saved_tensors
        y = x.mean(dim=1)
        x_proj = x + (z_target - y).unsqueeze(1)
        return process_node_target(x, x_proj)


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

        N, C, H_in, W_in = input.shape
        _, _, H_out, W_out = z_target.shape

        kH = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
        kW = kernel_size[1] if isinstance(kernel_size, tuple) else kernel_size

        # 1. Extract patches using unfold
        # Shape: (N, C * kH * kW, H_out * W_out)
        patches = F.unfold(input, kernel_size, stride=stride, padding=padding)
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

        
        return a_proj, None, None, None


def extract_patches(input, kernel_size, stride, padding) -> torch.Tensor:
    """Unfolds inputs into spatial patches and permutes for dense layers."""
    patches = F.unfold(input, kernel_size, dilation=1, padding=padding, stride=stride)
    
    H_out = (input.shape[2] + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
    W_out = (input.shape[3] + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
    
    # Reshape to (N, C*kH*kW, H_out, W_out) then permute to (N, H_out, W_out, C*kH*kW)
    return patches.view(input.shape[0], -1, H_out, W_out).permute(0, 2, 3, 1)


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
        
        target_img = target_sum / torch.clamp(overlap_counts, min=1.0)
        
        target = process_node_target(input, target_img)
        
        return target, None, None, None


@torch.compile(dynamic=True)
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
    """
    @staticmethod
    def forward(ctx, x, eps=1e-5):
        ctx.save_for_backward(x)
        return F.rms_norm(x, (x.size(-1),), eps=eps)

    @staticmethod
    def backward(ctx, z_target):
        x, = ctx.saved_tensors
        n = x.size(-1)
        sigma_bar = None

        if config.rmsnorm_backward_mode == "exact":
            x_bar = rmsnorm_proj_exact(
                x, z_target, config.rmsnorm_g, config.rmsnorm_newton_steps
            )
        else:
            # Step 1: Output projection
            # \bar{z} = \sqrt{n} z^+ / ||z^+||_2
            z_target_norm = torch.linalg.norm(z_target, dim=-1, keepdim=True)
            z_bar = math.sqrt(n) * (z_target / (z_target_norm + 1e-8))

            # Step 2: Input projection
            # \bar{\sigma} = mean(x * \bar{z})
            sigma_bar = (x * z_bar).mean(dim=-1, keepdim=True)

            # \bar{x} = \bar{\sigma} \bar{z}
            x_bar = sigma_bar * z_bar

        if DIAG is not None:
            stats = {
                "rel_delta": float((x_bar - x).norm() / (x.norm() + 1e-12)),
            }
            if sigma_bar is not None:
                stats["sigma_flip"] = float((sigma_bar < 0).float().mean())
            DIAG.record("rmsnorm", **stats)

        return process_node_target(x, x_bar), None



class Conversion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input.clone())
        return input.clone()
    @staticmethod
    def backward(ctx, z_target):
        (input,) = ctx.saved_tensors
        # Convert projection target into gradient: push input toward target
        grad = (input - z_target)
        #grad = grad/ torch.norm(grad)
        return grad

