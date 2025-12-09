"""Core projection operators for primitive functions.

Each primitive function has two components:
    ``operation(input_args, /) -> output``
    ``projection(input_args, output, /) -> projected_inputs``

The operation computes the forward pass, while the projection operator
computes the orthogonal projection onto the function's graph.
"""

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
    
    return (a_k[k][jnp.argsort(idx)].astype(jnp.complex64),)


max = make_computation("max", max_op, max_proj)

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


haddamarmul = make_computation("haddamarmul", haddamarmul_op, haddamarmul_proj)

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
    a = a.astype(jnp.float32)
    b = b.astype(jnp.float32)
    z = z.astype(jnp.float32)
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
    a_new = a_new.astype(jnp.complex64)
    b_new = b_new.astype(jnp.complex64)
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
