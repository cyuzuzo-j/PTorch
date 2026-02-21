"""Core projection operators for primitive functions.

Each primitive function has two components:
    ``operation(input_args, /) -> output``
    ``projection(input_args, output, /) -> projected_inputs``

The operation computes the forward pass, while the projection operator
computes the orthogonal projection onto the function's graph.
"""

import builtins
import jax
from jax import numpy as jnp
from jax import jacrev
from .. import config
from .computation import make_computation

def identity_op(a, /):
    """Identity operation returning input unchanged."""
    return a


def identity_proj(a, z, /):
    """Project onto identity function graph."""
    mid = (a + z) / 2
    return (mid,)


identity = make_computation("identity", identity_op, identity_proj)


def sum_op(a, /):
    """Sum operation for 1D arrays."""
    assert a.ndim == 1
    return a.sum()


def sum_proj(a, z, /):
    """Project onto sum function graph."""
    assert a.ndim == 1 and z.ndim == 0
    t = (z - a.sum()) / (a.size + z.size)
    return (a + t,)


sum_ = make_computation("sum", sum_op, sum_proj)

def _resolve_pool_padding(input_shape, pool_size, strides, padding):
    if isinstance(padding, str):
        if padding.upper() == "VALID":
            return (0, 0), (0, 0)
        if padding.upper() != "SAME":
            raise ValueError(f"Unsupported padding: {padding}")
        in_h, in_w = input_shape
        ph, pw = pool_size
        sh, sw = strides
        out_h = int(jnp.ceil(in_h / sh))
        out_w = int(jnp.ceil(in_w / sw))
        pad_h = builtins.max(0, (out_h - 1) * sh + ph - in_h)
        pad_w = builtins.max(0, (out_w - 1) * sw + pw - in_w)
        pad_top = pad_h // 2
        pad_left = pad_w // 2
        pad_bottom = pad_h - pad_top
        pad_right = pad_w - pad_left
        return (pad_top, pad_bottom), (pad_left, pad_right)
    return padding


def maxpool_op(a, /, *, pool_size=(2, 2), strides=(2, 2), padding="VALID"):
    """MaxPool operation for 4D arrays with shape (batch, height, width, channels)."""
    assert a.ndim == 4
    window = (1, pool_size[0], pool_size[1], 1)
    window_strides = (1, strides[0], strides[1], 1)
    return jax.lax.reduce_window(a, -jnp.inf, jax.lax.max, window, window_strides, padding)


def maxpool_proj(a, z, /, *, pool_size=(2, 2), strides=(2, 2), padding="VALID"):
    """Project onto max pooling function graph for non-overlapping windows."""
    assert a.ndim == 4 and z.ndim == 4
    if tuple(pool_size) != tuple(strides):
        raise ValueError("maxpool projection requires strides == pool_size")
    n, h_in, w_in, c = a.shape
    ph, pw = pool_size
    sh, sw = strides
    (pad_top, pad_bottom), (pad_left, pad_right) = _resolve_pool_padding(
        (h_in, w_in), pool_size, strides, padding
    )
    h_pad = h_in + pad_top + pad_bottom
    w_pad = w_in + pad_left + pad_right
    patches = jax.lax.conv_general_dilated_patches(
        a,
        filter_shape=pool_size,
        window_strides=strides,
        padding=padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    h_out, w_out = patches.shape[1], patches.shape[2]
    patches = patches.reshape(n, h_out, w_out, c, ph * pw)
    flat_patches = patches.reshape(-1, ph * pw)
    flat_z = z.reshape(-1)
    proj_flat = jax.vmap(lambda a_vec, z_scalar: max_proj(a_vec, z_scalar)[0])(
        flat_patches, flat_z
    )
    proj_patches = proj_flat.reshape(n, h_out, w_out, c, ph, pw)
    proj_patches = proj_patches.transpose(0, 1, 4, 2, 5, 3)
    proj_cover = proj_patches.reshape(n, h_out * ph, w_out * pw, c)
    h_cov = min(proj_cover.shape[1], h_pad)
    w_cov = min(proj_cover.shape[2], w_pad)
    proj_cover = proj_cover[:, :h_cov, :w_cov, :]
    pad_h = h_pad - h_cov
    pad_w = w_pad - w_cov
    if pad_h or pad_w:
        proj_cover = jnp.pad(
            proj_cover,
            ((0, 0), (0, pad_h), (0, pad_w), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )
    proj_input = proj_cover[:, pad_top : pad_top + h_in, pad_left : pad_left + w_in, :]
    covered_pad = jnp.zeros((1, h_pad, w_pad, 1), dtype=bool)
    covered_pad = covered_pad.at[:, :h_cov, :w_cov, :].set(True)
    covered = covered_pad[:, pad_top : pad_top + h_in, pad_left : pad_left + w_in, :]
    a_proj = jnp.where(covered, proj_input, a)
    return (a_proj,)

def max_op(a, /):
    """Maximum operation for 1D arrays."""
    assert a.ndim == 1
    return jnp.max(a)


def max_proj(a, z, /):
    """Project onto maximum function graph."""
    assert a.ndim == 1 and z.ndim == 0
    a = a.astype(jnp.float32)
    z = z.astype(jnp.float32)
    n = a.size

    # sort array
    idx = jnp.argsort(a)
    a_sorted = jnp.array(a)[idx]

    # compute candidate maxima
    z_k_flipped = (jnp.cumsum(jnp.flip(a_sorted)) + z) / jnp.arange(2, n + 2)
    z_k = jnp.flip(z_k_flipped)

    # compute candidate arrays
    i_ge_k = jnp.triu(jnp.ones((n, n), dtype=bool))
    a_k = jnp.where(i_ge_k, z_k[:, None], a_sorted[None, :])

    # compute distances
    dist = ((a_k - a_sorted) ** 2).sum(axis=1) + (z_k - z) ** 2

    # select valid candidates
    dist_valid = jnp.where(jnp.max(a_k, axis=1) <= z_k, dist, jnp.inf)

    # select candidate minimizing distance
    k = jnp.argmin(dist_valid)
    
    return (a_k[k][jnp.argsort(idx)].astype(jnp.bfloat16),)


max = make_computation("max", max_op, max_proj)

maxpool = make_computation("maxpool", maxpool_op, maxpool_proj)


def _solve_reduced_system(a_val, b_val, y_val):
    """
    Solves the projection using the reduced 2x2 system (Real/Imag of L)
    with precomputed algebraic invariants and explicit 2x2 inversion.
    """
    
    # --- Optimization 1: Precompute Constants ---
    # Numerator expansion: (a - L*conj(b))(b - L*conj(a)) 
    # = ab - L(|a|^2 + |b|^2) + L^2*conj(ab)
    # This reduces loop arithmetic significantly.
    
    P = a_val * b_val                # Product
    S = jnp.abs(a_val)**2 + jnp.abs(b_val)**2  # Sum of magnitudes
    P_conj = jnp.conj(P)
    
    def residual(l_vec):
        L = l_vec[0] + 1j * l_vec[1]
        
        # Denominator term
        L_mag_sq = jnp.abs(L)**2
        denom = (1.0 - L_mag_sq)
        
        # Optimized Numerator using invariants
        # P - L*S + L^2*conj(P)
        numerator = P - L * S + (L**2) * P_conj
        
        lhs = numerator / (denom**2 + 1e-8)
        rhs = y_val + L
        
        diff = lhs - rhs
        return jnp.array([jnp.real(diff), jnp.imag(diff)])

    J_fun = jax.jacfwd(residual)

    def cond_fun(state):
        l_vec, step_norm, iter_num = state
        return (step_norm > 1e-5) & (iter_num < 50)

    def body_fun(state):
        l_vec, _, iter_num = state
        
        R = residual(l_vec)
        J = J_fun(l_vec)
        
        # --- Optimization 2: Explicit 2x2 Linear Solve ---
        # Solve J * delta = -R using Cramer's rule
        # J = [[a, b], [c, d]]
        # det = ad - bc
        # inv = 1/det * [[d, -b], [-c, a]]
        
        # Unpack Jacobian elements for clarity
        j00, j01 = J[0, 0], J[0, 1]
        j10, j11 = J[1, 0], J[1, 1]
        
        det = j00 * j11 - j01 * j10
        inv_det = 1.0 / (det + 1e-12) # epsilon for safety
        
        # delta = -inv * R
        # d0 = -(J[1,1]*R[0] - J[0,1]*R[1]) / det
        # d1 = -(-J[1,0]*R[0] + J[0,0]*R[1]) / det
        
        d0 = (j01 * R[1] - j11 * R[0]) * inv_det
        d1 = (j10 * R[0] - j00 * R[1]) * inv_det
        
        delta = jnp.array([d0, d1])
        
        return l_vec + delta, jnp.linalg.norm(delta), iter_num + 1

    init_state = (jnp.zeros(2), 1.0, 0)
    final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)
    
    L_final_vec = final_state[0]
    L = L_final_vec[0] + 1j * L_final_vec[1]
    
    # Reconstruct primitives using the same denominators
    denom = 1.0 - jnp.abs(L)**2
    a_new = (a_val - L * jnp.conj(b_val)) / denom
    b_new = (b_val - L * jnp.conj(a_val)) / denom
    
    return a_new, b_new

def haddamarmul_proj_optimized(a, b, y):
    """
    Project onto Hadamard product graph using reduced analytical derivation.
    Input: a, b, y (Complex JAX Arrays of same shape)
    """
    if a.shape != b.shape or a.shape != y.shape:
        raise ValueError("All inputs must have the same shape")

    return jax.vmap(_solve_reduced_system)(a, b, y)


def _haddamarmul_F(z, const):
    w_r, w_i, x_r, x_i, y_r, y_i = const
    w_rp, w_ip, x_rp, x_ip, y_rp, y_ip, lam_r, lam_i = z

    return jnp.array([
        2*(w_rp - w_r) + lam_i * x_ip + lam_r * x_rp,
        2*(w_ip - w_i) - lam_r * x_ip + lam_i * x_rp,
        2*(x_rp - x_r) + lam_i * w_ip + lam_r * w_rp,
        2*(x_ip - x_i) - lam_r * w_ip + lam_i * w_rp,
        2*(y_rp - y_r) - lam_r,
        2*(y_ip - y_i) - lam_i,
        w_rp * x_rp - w_ip * x_ip - y_rp,
        w_rp * x_ip + w_ip * x_rp - y_ip
    ])

_haddamarmul_JF = jacrev(_haddamarmul_F)

def _haddamarmul_solve_single(const):
    z = jnp.zeros(8)
    z = z.at[0:6].set(const[0:6])  # initialize w_rp, w_ip, x_rp, x_ip, y_rp, y_ip
    
    def cond_fun(state):
        z, step_norm, iter_num = state
        return (step_norm > 1e-4) & (iter_num < 150)

    def body_fun(state):
        z, _, iter_num = state
        J = _haddamarmul_JF(z, const)
        f = _haddamarmul_F(z, const)
        delta = jnp.linalg.solve(J, -f)
        step_norm = jnp.linalg.norm(delta)
        return z + delta, step_norm, iter_num + 1

    init_state = (z, 1.0, 0)
    final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)
    return final_state[0]

def haddamarmul_op(a, b, /):
    """Haddamar product operation for 1D arrays."""
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("haddamarmul requires 1D arrays.")
    if a.size != b.size:
        raise ValueError(f"haddamarmul requires arrays of the same size. Got {a.size} and {b.size}.")
    return a * b

def haddamarmul_proj(a, b, y, /):
    """Project onto Hadamard product graph."""
    #jax.debug.print("haddamarmul_proj called with a {}, b {}, y {}", a.shape, b.shape, y.shape)
    consts = jnp.stack([
        jnp.real(a), jnp.imag(a),
        jnp.real(b), jnp.imag(b),
        jnp.real(y), jnp.imag(y)
    ], axis=-1)
    
    z_sol = jax.vmap(_haddamarmul_solve_single)(consts)
    
    w_rp = z_sol[:, 0]
    w_ip = z_sol[:, 1]
    x_rp = z_sol[:, 2]
    x_ip = z_sol[:, 3]
    
    a_new = w_rp + 1j * w_ip
    b_new = x_rp + 1j * x_ip
    return a_new, b_new

def real_proj(orig, x, /):
    """Project onto real function graph."""
    #jax.debug.print("projection to real called with orig {} x {} ", orig, x)
    x_real = jnp.real(x)
    return (x_real + 0j,)


haddamarmul = make_computation("haddamarmul", haddamarmul_op, haddamarmul_proj_optimized)

real = make_computation(
    "real",
    lambda x: jnp.real(x),
    lambda orig, x: real_proj(orig, x)
)

def dotproduct_op(a, b, /):
    """Dot product operation for 1D arrays."""
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("dotproduct requires 1D arrays.")
    if a.size != b.size:
        raise ValueError(f"dotproduct requires arrays of the same size. Got {a.size} and {b.size}.")
    return a.dot(b)


def bilinear_proj(a, b, z, /):
    """Project onto bilinear function graph using Newton's method."""
    a = a.astype(jnp.bfloat16)
    b = b.astype(jnp.bfloat16)
    z = z.astype(jnp.bfloat16)
    p = a @ b
    q = a @ a + b @ b

    def f(t):
        return ((1 + t**2) * p + t * q) / (1 - t**2) ** 2 - z + t

    f_prime = jax.grad(f)

    def newton_step(t):
        return t - f(t) / f_prime(t)

    
    t = jax.lax.fori_loop(0, config.bilinear_projection_num_newton_steps, lambda _, t: newton_step(t), 0.0, unroll=True)

    a_new = (a + t * b) / (1 - t**2)
    b_new = (b + t * a) / (1 - t**2)
    a_new = a_new.astype(jnp.bfloat16)
    b_new = b_new.astype(jnp.bfloat16)
    return a_new, b_new


dot = make_computation("dot", dotproduct_op, bilinear_proj)

def sum_step_activation_op(*args):
    """Sum inputs and apply step activation."""
    sum_args = sum(args)
    return jnp.where(sum_args >= 0, 1, -1)

def hermitian_symetry(a, /):
    """Enforce Hermitian symmetry on a 2D complex array."""
    a_conj_flip = jnp.conj(jnp.flip(jnp.flip(a, axis=0), axis=1))
    a_sym = (a + a_conj_flip) / 2
    return a_sym

def hermitian_symetry_proj(X):
    """
    Project a 2D complex array X onto the Hermitian-symmetric set
    that ensures a real 2D IFFT.
    """
    M, N = X.shape
    # conjugate + 180-degree rotation (flip both axes)
    X_flip_conj = jnp.conj(jnp.flip(jnp.flip(X, axis=0), axis=1))
    X_sym = 0.5 * (X + X_flip_conj)

    # find indices that are their own partner: 2u % M == 0 and 2v % N == 0
    u_idxs = jnp.arange(M)
    v_idxs = jnp.arange(N)
    u_self = (2 * u_idxs) % M == 0
    v_self = (2 * v_idxs) % N == 0

    # set imaginary part to zero at the "self-symmetric" grid points
    for u in jnp.where(u_self)[0]:
        for v in jnp.where(v_self)[0]:
            X_sym[u, v] = jnp.real(X_sym[u, v])  # force real

    return X_sym

def step_activation_proj(*args):
    """Project onto step activation function constraint: y = step(x) where step(x) = 1 if x >= 0, -1 otherwise."""
    inputs, output = args[:-1], args[-1]
    s = sum(inputs)
    
    # Element-wise projection: adjust inputs so their sum matches target output
    # For each element, if s >= 0 but output == -1, or s < 0 but output == 1, we need to adjust
    mid = (s - output) / (len(inputs) + 1)
    
    # Adjust inputs: move them toward the midpoint
    projected_inputs = tuple(x - mid for x in inputs)
    
    return projected_inputs

step = make_computation("step_activation", sum_step_activation_op, step_activation_proj)

def sum_relu_op(*args):
    """Sum inputs and apply ReLU activation."""
    return jax.nn.relu(sum(args))    

def sum_relu_proj(*args):
    """Project onto sum-ReLU function graph."""
    inputs, output = args[:-1], args[-1]

    # compute intermediate values
    s = sum(inputs)
    mid_1 = s / len(inputs)
    mid_2 = (output - s) / (len(inputs) + 1)

    # solution 1
    new_inputs_1 = [jnp.where(s < 0, x, x - mid_1) for x in inputs]
    dist_1 = jnp.where(s < 0, output**2, output**2 + len(inputs) * mid_1**2)

    # solution 2
    new_output_2 = jnp.maximum(0, output - mid_2)
    new_inputs_2 = [jnp.where(new_output_2 == 0, x - mid_1, x + mid_2) for x in inputs]
    dist_2 = jnp.where(output == 0, output**2 + len(inputs) * mid_1**2, (len(inputs) + 1) * mid_2**2)

    # select solution minimizing the distance
    return tuple(jnp.where(dist_1 < dist_2, x_1, x_2) for x_1, x_2 in zip(new_inputs_1, new_inputs_2))


sum_relu = make_computation("sum_relu", sum_relu_op, sum_relu_proj)


def quantize_op(a, /, *, levels=2, scale=1.0):
    """Quantize values to discrete levels."""
    assert levels >= 2
    step = 2 * scale / (levels - 1)
    z = jnp.round((a + scale) / step)
    z = jnp.clip(z, 0, levels - 1)
    z = z * step - scale
    return z


def quantize_proj(a, z, /, *, levels=2, scale=1.0):
    """Project onto quantization function graph."""

    quantization_points = jnp.linspace(-scale, scale, levels)
    midpoints = (quantization_points[:-1] + quantization_points[1:]) / 2
    lower_bounds = jnp.concatenate([jnp.array([-jnp.inf]), midpoints])
    upper_bounds = jnp.concatenate([midpoints, jnp.array([jnp.inf])])

    a_candidates = jnp.select(
        [a[..., None] < lower_bounds, a[..., None] > upper_bounds], [lower_bounds, upper_bounds], a[..., None]
    )

    z_candidates = jnp.broadcast_to(quantization_points, a_candidates.shape)

    dist = (a_candidates - a[..., None]) ** 2 + (z_candidates - z[..., None]) ** 2

    min_idx = jnp.argmin(dist, axis=-1, keepdims=True)
    a_new = jnp.take_along_axis(a_candidates, min_idx, axis=-1).squeeze(-1)
    return (a_new,)


quantize = make_computation("quantize", quantize_op, quantize_proj)


def margin_loss_op(logits, labels, /):
    """Margin loss operation."""
    return logits


def margin_loss_proj(logits, labels, *args):
    """Project onto margin loss constraint."""

    cond_0 = (labels <= 0) & (logits > 0)
    cond_1 = (labels > 0) & (logits < labels)

    new_logits = jnp.select([cond_0, cond_1], [0.0, labels], logits)

    return new_logits, labels


margin_loss = make_computation("margin_loss", margin_loss_op, margin_loss_proj)

def mean_squared_op(predictions, targets, /):
    """Mean squared error operation."""
    squared_errors=  jnp.abs(predictions - targets) ** 2
    return jnp.mean(squared_errors)

def mse_prox(predictions, targets, idk, /):
    """Project onto mean squared error constraint. """
    out = (predictions+ targets) /2
    #jax.debug.print("preds {} , targets,{}, idk {}", predictions, targets, idk )
    return out, out

mse = make_computation("mse_loss", mean_squared_op, mse_prox)

def reparameterize_op(*args):
    mu = args[0]
    sigma = args[1]
    #jax.debug.print("Reparameterize mu shape:{}, sigma shape:{}, arglen {}", mu.shape, sigma.shape, args)
    """Reparameterization operation."""
    eps = jax.random.normal(jax.random.PRNGKey(0), shape=mu.shape, ).astype(mu.dtype)
    #jax.debug.print("eps shape: {}", eps.shape)
    out = mu + sigma * eps
    #jax.debug.print("reparameterize output shape: {}", out.shape)
    return out

def project_to_normal_vec(mu0, sigma0, z0, lam=1.0, tol=1e-8, maxiter=200):
    """
    Minimize for each dimension i:
        (z_i - z0_i)^2 + (μ_i - μ0_i)^2 + (σ_i - σ0_i)^2
        + λ * [ (z_i - μ_i)^2 / σ_i^2 + log σ_i^2 ]
    so that z_i resembles a Normal(μ_i, σ_i^2), while staying close to priors.
    """
    # elementwise equation for σ_i and its derivative
    def sigma_fun(sigma, z0, mu0, sigma0, lam):
        return (
            (sigma - sigma0)
            - lam * (sigma * (z0 - mu0) ** 2) / (2 * lam + sigma**2) ** 2
            + lam / sigma
        )

    def sigma_fun_derivative(sigma, z0, mu0, sigma0, lam):
        diff_sq = (z0 - mu0) ** 2
        denom = 2 * lam + sigma**2
        term1 = 1.0
        term2 = -lam * (diff_sq * denom**2 - sigma * diff_sq * 2 * sigma * 2 * denom) / (denom**4)
        term3 = -lam / (sigma**2)
        return term1 + term2 + term3

    # Newton's method for each dimension
    def solve_sigma_single(z0_i, mu0_i, sigma0_i):
        sigma = jnp.maximum(sigma0_i, 1e-6)
        
        def newton_body(sigma):
            f_val = sigma_fun(sigma, z0_i, mu0_i, sigma0_i, lam)
            f_prime = sigma_fun_derivative(sigma, z0_i, mu0_i, sigma0_i, lam)
            sigma_new = sigma - f_val / (f_prime + 1e-8)
            return jnp.maximum(sigma_new, 1e-6)
        
        sigma = jax.lax.fori_loop(0, maxiter, lambda _, s: newton_body(s), sigma)
        return jnp.maximum(sigma, 1e-8)
    
    # vectorize over dimensions
    solve_sigma_vmap = jax.vmap(solve_sigma_single, in_axes=(0, 0, 0))
    sigma = solve_sigma_vmap(z0, mu0, sigma0)

    # compute z*, mu* elementwise
    denom = 2 * lam + sigma**2
    z = (lam * mu0 + lam * z0 + sigma**2 * z0) / denom
    mu = (lam * mu0 + lam * z0 + mu0 * sigma**2) / denom

    return mu, sigma

reparameterize = make_computation("reparameterize", reparameterize_op, project_to_normal_vec)

#kl_divergence = make_computation("kl_divergence", kl_divergence_op, prox_kl_std_normal)
def cross_entropy_op(logits, labels, /):
    """Cross-entropy operation."""
    return logits


def corss_entropy_prox(logits, labels, _, /):
    """Project onto cross-entropy constraint using iterative methods."""

    method = config.cross_entropy_method
    lmbda = config.cross_entropy_lambda
    steps = config.cross_entropy_num_steps

    if method == "newton":

        def newton(logits, labels):
            s = jax.nn.softmax(logits)
            x = logits
            for _ in range(steps):
                G = (x - logits) / lmbda + s - labels
                H = jnp.eye(x.size) / lmbda + jnp.diag(s) - jnp.outer(s, s)
                dx = jax.scipy.linalg.solve(H, G, assume_a="pos")
                x = x - dx
            return x

        x = jax.vmap(newton)(logits, labels)

    elif method == "fixed_point":
        x = logits
        for _ in range(steps):
            x = logits + lmbda * (labels - jax.nn.softmax(x))

    else:
        raise ValueError(f"Unknown method: {method}")

    return x, labels


cross_entropy = make_computation("cross_entropy", cross_entropy_op, corss_entropy_prox)

def matmul_op(a, b, /):
    """Matrix multiplication operation."""
    return jnp.matmul(a, b)
    
def matmul_proj(a, b, z, /):
    """Project onto matrix multiplication constraint."""
    # helper to compute projection for a single matrix multiplication
    # (M, K) @ (K, N) = (M, N)
    def project_2d(a, b, z):
        # We handle this by vmapping the bilinear projection logic over the M and N dimensions
        # a: (M, K), b: (K, N), z: (M, N)
        
        # Expand dims to broadcast against each other
        # a_exp: (M, 1, K)
        # b_exp: (1, N, K) (transpose b for easier dot product alignment if using bilinear_proj logic)
        # But wait, bilinear_proj assumes dot product a.b = z.
        # matmul element z_ij = row_i(a) . col_j(b)
        
        M, K = a.shape
        N = b.shape[1]
        
        # Prepare inputs for M*N dot product projections
        # We want to project (row_i, col_j, z_ij) -> (row_i_new, col_j_new) for all i, j
        
        a_rows = a[:, None, :]  # (M, 1, K)
        b_cols = b.T[None, :, :] # (1, N, K). Note b is (K, N), so b.T is (N, K).
        z_vals = z[:, :, None] # (M, N, 1) to match dimensionality if needed, or just (M, N)
        
        # we can use vectorized bilinear_proj
        # bilinear_proj takes (K,), (K,), () -> (K,), (K,)
        # We map over M and N.
        
        # vmap over N (cols of b)
        # vmap over M (rows of a)
        # vmap over K is inside bilinear_proj (dot product)
        
        # bilinear_proj_vmapped = jax.vmap(jax.vmap(bilinear_proj, in_axes=(None, 0, 0)), in_axes=(0, None, 0))
        # Wait, if we use in_axes=(None, 0, 0) for the inner vmap (over N):
        #   a (one row) is shared (None).
        #   b (all cols) are mapped (0).
        #   z (row of scalars) is mapped (0).
        # Output: (N, K), (N, K) -> updates for the row of A, updates for all cols of B.
        
        # Outer vmap (over M):
        #   a (all rows) are mapped (0).
        #   b (cols) are shared (None) -> actually passed as full array
        #   z (matrix) is mapped (0).
        
        project_single = bilinear_proj
        
        # Map over cols of B (and elements of Z row)
        # input: a_row (K,), b_cols (N, K), z_row (N,)
        # output: a_row_updates (N, K), b_cols_updates (N, K)
        project_row = jax.vmap(project_single, in_axes=(None, 1, 0)) 
        
        # Map over rows of A (and rows of Z)
        # input: a (M, K), b (K, N), z (M, N)
        # output: a_updates (M, N, K), b_updates (M, N, K)
        project_matrix = jax.vmap(project_row, in_axes=(0, None, 0))
        
        a_updates, b_updates = project_matrix(a, b, z)
        
        # a is updated N times (once for each col of b). Average these.
        a_new = jnp.mean(a_updates, axis=1) # (M, K)
        
        # b is updated M times (once for each row of a). Average these.
        # b_updates is (M, N, K). The output from project_row for 'b' was (N, K) which corresponds to b.T.
        # Wait, project_single returns (K,), (K,).
        # project_row returns (N, K), (N, K).
        # project_matrix returns (M, N, K), (M, N, K).
        # The second output corresponds to 'b'. In project_row, we passed b as (K, N) but used in_axes=1.
        # So b_updates[i, j, :] is the update for col j of b, from row i of a.
        # We need to average over i (rows of a) to get the update for col j.
        # AND we need to transpose it back to (K, N) because b_updates stored it as (K,) vectors.
        # b_updates shape is (M, N, K).
        # Average over M: (N, K).
        # Transpose to (K, N).
        
        b_new = jnp.mean(b_updates, axis=0).T
        
        return a_new, b_new
    # Detect batch dimensions
    # Current limitation: supports simple broadcasting where 'a' has batch dims and 'b' does not, or vice versa?
    # Or strict 'a' is broadcasted?
    # User's efficient matmul request implied broadcasting.
    # Typically: a is (B, M, K), b is (K, N). z is (B, M, N).
    
    a_ndim = a.ndim
    b_ndim = b.ndim
    z_ndim = z.ndim
    
    # Assume standard matmul broadcasting rules: 
    # Last 2 dims are matrix dims. Leading dims are batch.
    # We only handle the case where one is broadcasted against the other for now, or simple repeats.
    
    if a_ndim > b_ndim:
        # Case: A has batch dims, B does not.
        # Flatten batch dims of A and Z for scanning.
        batch_shape = a.shape[:-2]
        M, K = a.shape[-2:]
        N = b.shape[-1]
        
        # Check shapes match expectation
        assert b.shape == (K, N)
        assert z.shape == batch_shape + (M, N)
        
        # Flatten batch dims
        a_flat = a.reshape(-1, M, K)
        z_flat = z.reshape(-1, M, N)
        num_batches = a_flat.shape[0]
        
        # Scan over batches to compute average update for b and individual updates for a
        # We use a recursive mean formula for b.
        # mean_n = mean_{n-1} + (x_n - mean_{n-1}) / n
        
        def scan_fn(carry, inputs):
            b_mean, n = carry
            a_i, z_i = inputs
            
            # Project current batch
            a_new_i, b_new_i = project_2d(a_i, b_mean, z_i) 
            # Note: technically we should project against the *original* b? 
            # Or the running mean b? 
            # In alternating projections, we project the *current estimate*.
            # But here 'b' is a shared parameter. 
            # The 'repeat_inverse' logic averages the projections of the SAME input 'b'.
            # So we should use 'b' (the input to this function) for all projections?
            # Yes, standard repeat_inverse takes 'z' (the outputs of the branches) and averages them.
            # Here, the 'branches' are the projections of (a_i, b, z_i).
            # So we should pass 'b' (the constant input from outside) to project_2d.
            # BUT, we want to return the average of the *outputs* (b_new_i).
            
            # Wait, the scan carry should be the running mean of the *updates*.
            # The input 'b' to project_2d should be the *original input b*.
            
            # We can't use 'b_mean' as input to project_2d because that would imply sequential updates (like SGD).
            # This is a projection operator, it's stateless within the step.
            # So we use 'b' from the outer scope.
            
            a_new_i, b_new_i = project_2d(a_i, b, z_i)
            
            # Update running mean of b_new
            # n is 1-based index (current count)
            n += 1
            b_mean_new = b_mean + (b_new_i - b_mean) / n
            
            return (b_mean_new, n), a_new_i
        # Initialize running mean with zeros or first element logic?
        # A simple trick is to start with 0 and count. 
        # But we need shape of b_new_i. It's same as b.
        
        init_carry = (jnp.zeros_like(b), 0)
        
        # Run scan
        (b_new, _), a_new_flat = jax.lax.scan(scan_fn, init_carry, (a_flat, z_flat))
        
        # Reshape a_new back
        a_new = a_new_flat.reshape(a.shape)
        
        return a_new, b_new
    elif b_ndim > a_ndim:
        # Symmetric case: B has batch dims, A does not.
        raise NotImplementedError("Broadcasting A (batch dims in B) not yet implemented in efficient matmul.")
    else:
        # Standard matrix mult (no broadcasting or same batch dims)
        # If same batch dims, we just vmap over them (no averaging needed).
        # If no batch dims, just project_2d.
        if a_ndim == 2:
             return project_2d(a, b, z)
        else:
             # Same batch dims: vmap project_2d over the batch dims
             # Treat all leading dims as batch
             # We need to flatten them typically or use vmap recursively/repeatedly?
             # jax.vmap handles arbitrary leading dims if mapped.
             # But inputs are a, b, z.
             # project_2d handles (M,K), (K,N), (M,N).
             # We can just vmap project_2d appropriate number of times or flatten.
             
             # Flatten
             batch_shape = a.shape[:-2]
             a_flat = a.reshape(-1, a.shape[-2], a.shape[-1])
             b_flat = b.reshape(-1, b.shape[-2], b.shape[-1])
             z_flat = z.reshape(-1, z.shape[-2], z.shape[-1])
             
             project_2d_vmapped = jax.vmap(project_2d)
             a_new_flat, b_new_flat = project_2d_vmapped(a_flat, b_flat, z_flat)
             
             return a_new_flat.reshape(a.shape), b_new_flat.reshape(b.shape)
matmul = make_computation("matmul", matmul_op, matmul_proj)
