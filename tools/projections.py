import jax
import jax.numpy as jnp
import jax.lax as lax
from functools import partial


# -------------------------------------------------------------------------
# Dyadic-rational utilities: ℛ = ℤ[1/2^p] = Δ·ℤ with Δ = 2^-p.
# -------------------------------------------------------------------------

def quantize_to_dyadic(x, p):
    """Snap x to the dyadic lattice ℛ = ℤ[1/2^p] (nearest)."""
    Delta = jnp.asarray(2.0, dtype=x.dtype) ** (-p)
    return jnp.round(x / Delta) * Delta


def _ceil_to_dyadic(x, p):
    Delta = jnp.asarray(2.0, dtype=x.dtype) ** (-p)
    return jnp.ceil(x / Delta) * Delta


@partial(jax.jit, static_argnames=("p", "q", "num_steps", "bump_steps"))
def bilinear_dyadic_linf(h, theta, h_plus, p=12, q=0, num_steps=40, bump_steps=8):
    """
    Discrete ℓ∞ bilinear projection over the dyadic ring ℛ = ℤ[1/2^p].

    Solves
        min_{h̄,θ̄∈ℛⁿ, h̄+∈ℛ} ‖(h̄,θ̄,γ²h̄+) - (h,θ,γ²h+)‖_∞
        s.t. |θ̄ᵀh̄ - h̄+| ≤ Δ/2,   γ² = 2^q,  Δ = 2^-p.

    Implementation follows the spec: safeguarded Newton on the piecewise-quadratic
    F(ε) = Σ_k M_k(ε) + ε/γ² - |h+ - D|, then ceiling-round to ℛ. The case h+ < D
    is reduced to h+ > D by flipping h → -h (which swaps a_k ↔ b_k).
    """
    h      = quantize_to_dyadic(h,      p)
    theta  = quantize_to_dyadic(theta,  p)
    h_plus = quantize_to_dyadic(h_plus, p)

    Delta      = jnp.asarray(2.0, dtype=h.dtype) ** (-p)
    inv_gamma2 = jnp.asarray(2.0, dtype=h.dtype) ** (-q)

    D          = jnp.dot(theta, h)
    target_gap = h_plus - D
    s          = jnp.sign(target_gap)

    h_eff = jnp.where(s < 0, -h, h)

    a   = jnp.abs(h_eff + theta)
    b   = jnp.abs(h_eff - theta)
    rhs = jnp.abs(target_gap)

    def F_and_dF(eps):
        m   = (2.0 * eps >= b - a).astype(eps.dtype)
        Mk  = m * (a * eps + eps * eps) + (1.0 - m) * (b * eps - eps * eps)
        dMk = m * (a + 2.0 * eps)        + (1.0 - m) * (b - 2.0 * eps)
        return jnp.sum(Mk) + eps * inv_gamma2 - rhs, jnp.sum(dMk) + inv_gamma2

    eps0 = rhs / (jnp.sum(jnp.maximum(a, b)) + inv_gamma2 + 1e-30)

    def newton_step(_, eps):
        F, dF = F_and_dF(eps)
        raw   = F / (dF + 1e-30)
        cap   = 0.5 * jnp.maximum(eps, Delta) + Delta
        step  = jnp.clip(raw, -cap, cap)
        return jnp.maximum(eps - step, 0.0)

    eps = lax.fori_loop(0, num_steps, newton_step, eps0)

    eps_star = _ceil_to_dyadic(eps, p)

    def bump(_, eps):
        F, _ = F_and_dF(eps)
        return jnp.where(F < 0.0, eps + Delta, eps)

    eps_star = lax.fori_loop(0, bump_steps, bump, eps_star)

    m_mask     = (2.0 * eps_star >= b - a).astype(eps_star.dtype)
    sign_plus  = jnp.where((h_eff + theta) >= 0.0,  1.0, -1.0)
    sign_minus = jnp.where((h_eff - theta) >= 0.0, 1.0, -1.0)

    dh_eff = m_mask * eps_star * sign_plus + (1.0 - m_mask) * eps_star * sign_minus
    dtheta = m_mask * eps_star * sign_plus + (1.0 - m_mask) * (-eps_star) * sign_minus

    dh = jnp.where(s < 0, -dh_eff, dh_eff)

    h_bar      = h     + dh
    theta_bar  = theta + dtheta
    h_plus_bar = quantize_to_dyadic(jnp.dot(theta_bar, h_bar), p)

    h_bar      = jnp.where(s == 0.0, h,      h_bar)
    theta_bar  = jnp.where(s == 0.0, theta,  theta_bar)
    h_plus_bar = jnp.where(s == 0.0, h_plus, h_plus_bar)
    return h_bar, theta_bar, h_plus_bar


@partial(jax.jit, static_argnames=("p",))
def matMul_dyadic_quantized(X, W, Z, alpha=1.0, g=1.0, omega=1.0, num_steps=10, p=12):
    """
    Continuous joint matMul projection followed by lattice quantization to ℛ = ℤ[1/2^p].

    Fixed-point analogue: solve the continuous L2 joint problem then snap outputs to
    the dyadic lattice. Constraint W̄@X̄ = Z̄ holds up to per-element residual O(2^-p).
    """
    X_p, W_p, Z_p = matMul(X, W, Z, alpha=alpha, g=g, omega=omega, num_steps=num_steps)
    return (quantize_to_dyadic(X_p, p),
            quantize_to_dyadic(W_p, p),
            quantize_to_dyadic(Z_p, p))


@partial(jax.jit, static_argnames=("p",))
def matMulfixedX_dyadic_quantized(X, W, Z, g=1.0, omega=1.0, p=12):
    """Lattice-quantized variant of matMulfixedX (X held fixed and *also* snapped)."""
    X_p, W_p, Z_p = matMulfixedX(X, W, Z, g=g, omega=omega)
    return (quantize_to_dyadic(X_p, p),
            quantize_to_dyadic(W_p, p),
            quantize_to_dyadic(Z_p, p))


# -------------------------------------------------------------------------
# Packed dyadic representation: store integer mantissa with implicit scale 2^-p.
# This is what actually saves memory — the float-level quantize_to_dyadic above
# only constrains the *value*, not the *encoding*.
# -------------------------------------------------------------------------

_INT_DTYPE = {8: jnp.int8, 16: jnp.int16, 32: jnp.int32}


def _int_range(bits):
    lo = -(1 << (bits - 1))
    hi =  (1 << (bits - 1)) - 1
    return lo, hi


@partial(jax.jit, static_argnames=("p", "bits"))
def pack_dyadic(x, p=12, bits=16):
    """
    Encode x as an int{bits} mantissa m with implicit scale Δ = 2^-p,
    so that the represented value is m·Δ. Saturates on overflow.
    """
    Delta = jnp.asarray(2.0, dtype=jnp.float32) ** (-p)
    m_real = jnp.round(x / Delta)
    lo, hi = _int_range(bits)
    m_clip = jnp.clip(m_real, lo, hi)
    return m_clip.astype(_INT_DTYPE[bits])


@partial(jax.jit, static_argnames=("p", "dtype"))
def unpack_dyadic(m, p=12, dtype=jnp.float32):
    """Decode packed integer mantissa to a float tensor: x = m · 2^-p."""
    Delta = jnp.asarray(2.0, dtype=dtype) ** (-p)
    return m.astype(dtype) * Delta


# --- Per-tensor (shared-exponent) dyadic ---------------------------------
# Same dyadic lattice Δ·ℤ with Δ = 2^-p, but p is chosen per tensor from its
# absmax instead of being a single global constant. The scale stays a power of
# two, so this is still dyadic — just one exponent per tensor (a block / shared-
# exponent format). Picking p by absmax fills the int{bits} range without ever
# saturating, which is what lets narrow widths (int8) carry a wide value range.

def dyadic_scale_pertensor(x, bits):
    """Per-tensor dyadic exponent: p = floor((bits-1) - log2(max|x|)), so the
    largest |value| lands just inside the int{bits} range. Returns a scalar."""
    amax = jnp.max(jnp.abs(x))
    return jnp.floor((bits - 1) - jnp.log2(jnp.maximum(amax, 1e-30)))


@partial(jax.jit, static_argnames=("bits",))
def pack_dyadic_dyn(x, p, bits):
    """pack_dyadic with a traced (data-dependent) exponent p; saturates on overflow."""
    Delta = jnp.exp2(-p)
    lo, hi = _int_range(bits)
    return jnp.clip(jnp.round(x / Delta), lo, hi).astype(_INT_DTYPE[bits])


@partial(jax.jit, static_argnames=("dtype",))
def unpack_dyadic_dyn(m, p, dtype=jnp.float32):
    """Decode a dynamic-exponent packed tensor: x = m · 2^-p."""
    return m.astype(dtype) * jnp.exp2(-p).astype(dtype)


@partial(jax.jit, static_argnames=("p", "bits"))
def matMul_dyadic_packed(X_int, W_int, Z_int,
                         alpha=1.0, g=1.0, omega=1.0, num_steps=10,
                         p=12, bits=16):
    """
    Joint matMul projection on packed dyadic tensors.

    Inputs are int{bits} mantissas (implicit scale 2^-p); outputs are likewise.
    The projection itself runs in float (decode → solve → re-encode), but
    *between calls* the tensors live in int{bits} memory — that's where the
    saving comes from.
    """
    X = unpack_dyadic(X_int, p); W = unpack_dyadic(W_int, p); Z = unpack_dyadic(Z_int, p)
    Xp, Wp, Zp = matMul(X, W, Z, alpha=alpha, g=g, omega=omega, num_steps=num_steps)
    return (pack_dyadic(Xp, p, bits),
            pack_dyadic(Wp, p, bits),
            pack_dyadic(Zp, p, bits))


@partial(jax.jit, static_argnames=("p", "bits"))
def matMulfixedX_dyadic_packed(X_int, W_int, Z_int, g=1.0, omega=1.0, p=12, bits=16):
    X = unpack_dyadic(X_int, p); W = unpack_dyadic(W_int, p); Z = unpack_dyadic(Z_int, p)
    Xp, Wp, Zp = matMulfixedX(X, W, Z, g=g, omega=omega)
    return (pack_dyadic(Xp, p, bits),
            pack_dyadic(Wp, p, bits),
            pack_dyadic(Zp, p, bits))


@jax.jit
def matMul(X, W, Z, alpha=1.0, g=1.0, omega=1.0, num_steps=10, residual=False):
    """
    Exact independent bilinear projection for W @ X = Z adapted from ptorch.
    Analytically optimized to avoid O(M*N*K) memory expansions. 
    Pointwise Newton method runs in O(M*N) with numerical safeguards.
    X: (dim_in, N)
    W: (M, dim_in)
    Z: (M, N)
    Returns projected X, W, Z matching input shapes.
    """
    # Enforce 2D arrays to prevent IndexError for 1D weights/vectors
    A = jnp.atleast_2d(W)
    B = jnp.atleast_2d(X)
    Z_target = jnp.atleast_2d(Z)

    M = A.shape[0]
    N = B.shape[1]

    # Use B directly (assuming residual=False for standard matmul)
    B_eff = B

    # 1. Compute pairwise operations in O(M*N)
    p = A @ B_eff  # (M, N)
    qa = jnp.sum(A * A, axis=-1, keepdims=True)  # (M, 1)
    qb = jnp.sum(B_eff * B_eff, axis=-2, keepdims=True)  # (1, N)
    q_eff = qa + alpha * qb  # (M, N)

    t = jnp.zeros_like(p)
    target_penalty = (omega ** 2) / (g ** 2)

    # --- NUMERICAL SAFEGUARDS ---
    eps = 1e-5             # Minimum distance from the singularity
    damping = 1e-4         # Levenberg-Marquardt damping factor
    max_step_size = 0.5    # Maximum allowable change in t per step
    
    max_t_val = jnp.sqrt(jnp.maximum(alpha - eps, eps))
    
    # 3. Newton's Method (fused pointwise ops, numerically stabilized)
    def newton_step(i, t_val):
        t_val = jnp.clip(t_val, a_min=-max_t_val, a_max=max_t_val)
        
        t2 = jnp.square(t_val)
        alpha_minus_t2 = alpha - t2 

        N_num = alpha * (p * (alpha + t2) + t_val * q_eff)
        f_val = (N_num / jnp.square(alpha_minus_t2)) - Z_target + t_val * target_penalty

        N_prime = alpha * (2.0 * t_val * p + q_eff)
        f_prime_val = ((N_prime * alpha_minus_t2) + 4.0 * t_val * N_num) / (alpha_minus_t2 ** 3) + target_penalty

        raw_step = f_val / (jnp.abs(f_prime_val) + damping)
        step = jnp.clip(raw_step, a_min=-max_step_size, a_max=max_step_size)
        
        return t_val - step

    t = jax.lax.fori_loop(0, num_steps, newton_step, t)

    # Final boundary clamp
    t = jnp.clip(t, a_min=-max_t_val, a_max=max_t_val)

    # 4. Analytical Consensus Reconstruction
    t2 = jnp.square(t)
    denom = alpha - t2
    inv_denom = 1.0 / denom       
    t_inv_denom = t / denom       

    sum_inv_denom_j = jnp.sum(inv_denom, axis=-1, keepdims=True) 
    A_proj = (alpha / N) * (A * sum_inv_denom_j + t_inv_denom @ B_eff.T)

    sum_inv_denom_i = jnp.sum(inv_denom, axis=-2, keepdims=True) 
    B_eff_proj = (1.0 / M) * (alpha * B_eff * sum_inv_denom_i + A.T @ t_inv_denom)

    Z_proj = Z_target - t * target_penalty

    # Return elements matching the original parameter order (X, W, Z) and shapes
    return B_eff_proj.reshape(X.shape), A_proj.reshape(W.shape), Z_proj.reshape(Z.shape)

@jax.jit
def matMulfixedX(X, W, Z, g=1.0, omega=1.0):
    """
    Exact projection onto {W, Z : W @ X = Z} with X held fixed.

    Derived from KKT conditions of:
        min  (1/2)||W - W0||^2 + (omega^2 / 2g^2)||Z - Z0||^2
        s.t. W @ X = Z

    Closed-form (no Newton iteration):
        S     = X^T @ X                            (N x N Gram matrix)
        P     = W0 @ X                             (M x N current product)
        Z_proj = (P + λ * Z0 @ S) @ (I + λ*S)^-1  (linear solve)
        T      = λ * (Z_proj - Z0)                 (dual variable)
        W_proj = W0 - T @ X^T                      (primal update)

    where λ = omega^2 / g^2.

    Args:
        X: (dim_in, N)  — FIXED, not updated
        W: (M, dim_in)
        Z: (M, N)

    Returns:
        X (unchanged), W_proj, Z_proj
    """
    A  = jnp.atleast_2d(W)
    B  = jnp.atleast_2d(X)
    Z0 = jnp.atleast_2d(Z)

    lam = (omega ** 2) / (g ** 2)   # scalar penalty  λ = ω²/g²

    # Precompute reused quantities
    P = A @ B          # (M, N)  — current W @ X
    S = B.T @ B        # (N, N)  — Gram matrix X^T X

    # Solve: Z_proj @ (I + λS) = P + λ Z0 S
    # Equivalently: (I + λS) Z_proj^T = (P + λ Z0 S)^T  [symmetric system]
    system_matrix = jnp.eye(S.shape[0]) + lam * S        # (N, N), symmetric PD
    rhs           = P + lam * (Z0 @ S)                    # (M, N)

    # jnp.linalg.solve(A, b) solves A @ x = b
    # We need x @ system_matrix = rhs  →  system_matrix @ x^T = rhs^T
    Z_proj = jnp.linalg.solve(system_matrix, rhs.T).T     # (M, N)

    # Recover dual variable T and project W
    T      = lam * (Z_proj - Z0)                          # (M, N)
    W_proj = A - T @ B.T                                  # (M, dim_in)

    return X.reshape(X.shape), W_proj.reshape(W.shape), Z_proj.reshape(Z.shape)

@jax.jit
def leakyRelu(x, W, z, slope=0.1):
    """
    Vectorized Leaky ReLU activation projection.
    x is the pre-activation input, W is the identity matrix, z is the target post-activation.
    """
    H = W @ x

    # Solution 1: project onto inactive branch (x <= 0, output = slope * x)
    x_1 = jnp.clip((H + slope * z) / (1.0 + slope * slope), a_max=0)
    y_1 = slope * x_1
    dist_1 = (H - x_1)**2 + (z - y_1)**2

    # Solution 2: project onto active branch (x > 0, output = x)
    x_2 = jnp.clip((H + z) / 2.0, a_min=0)
    y_2 = x_2
    dist_2 = (H - x_2)**2 + (z - y_2)**2

    # Select solution minimizing distance
    new_H = jnp.where(dist_1 < dist_2, x_1, x_2)
    new_z = jnp.where(dist_1 < dist_2, y_1, y_2)

    return new_H, W, new_z

@jax.jit
def classifierOutput(x, w, y, delta=1):
    """ project onto classification output constraints y >= delta for positive class, y <= 0 for negative class"""
    x_new = jnp.where(y == 1, jnp.maximum(x, delta), jnp.minimum(x, 0))
    return x_new, w, y
