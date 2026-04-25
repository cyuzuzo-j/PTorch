import jax
import jax.numpy as jnp
import math

import functools

@functools.partial(jax.jit, static_argnames=['num_steps', 'residual'])
def matmul_proj(A, B, Z, t_init=None, alpha=1.0, g=1.0, omega=1.0, num_steps=10, residual=False):
    """
    Exact independent bilinear projection ported from ptorch.
    A @ B = Z
    A: (M, K)
    B: (K, N)
    Z: (M, N)
    """
    M = A.shape[-2]
    N = B.shape[-1]

    if residual:
        I = jnp.eye(B.shape[-2], B.shape[-1])
        B_eff = I - B
    else:
        B_eff = B

    p = A @ B_eff  # (M, N)
    qa = jnp.sum(A * A, axis=-1, keepdims=True)  # (M, 1)
    qb = jnp.sum(B_eff * B_eff, axis=-2, keepdims=True)  # (1, N)
    q_eff = qa + alpha * qb  # (M, N)

    if t_init is not None and t_init.shape == p.shape:
        t = t_init
    else:
        t = jnp.zeros_like(p)

    target_penalty = (omega ** 2) / (g ** 2)

    eps = 1e-5
    damping = 1e-4
    max_step_size = 0.5
    
    max_t_val = math.sqrt(max(alpha - eps, eps))

    def step_fn(t):
        t = jnp.clip(t, -max_t_val, max_t_val)
        
        t2 = jnp.square(t)
        alpha_minus_t2 = alpha - t2

        N_num = alpha * (p * (alpha + t2) + t * q_eff)
        f_val = (N_num / jnp.square(alpha_minus_t2)) - Z + t * target_penalty

        N_prime = alpha * (2.0 * t * p + q_eff)
        f_prime_val = ((N_prime * alpha_minus_t2) + 4.0 * t * N_num) / (jnp.power(alpha_minus_t2, 3)) + target_penalty

        raw_step = f_val / (jnp.abs(f_prime_val) + damping)
        step = jnp.clip(raw_step, -max_step_size, max_step_size)
        
        return t - step

    # unroll the fixed number of steps
    for _ in range(num_steps):
        t = step_fn(t)

    t = jnp.clip(t, -max_t_val, max_t_val)

    t2 = jnp.square(t)
    denom = alpha - t2
    inv_denom = 1.0 / denom       # (M, N)
    t_inv_denom = t / denom       # (M, N)

    sum_inv_denom_j = jnp.sum(inv_denom, axis=-1, keepdims=True) # (M, 1)
    
    # A_proj analytically averages over N proposals via Matrix Math
    A_proj = (alpha / N) * (A * sum_inv_denom_j + t_inv_denom @ jnp.swapaxes(B_eff, -2, -1))

    # B_proj analytically averages over M proposals via Matrix Math
    sum_inv_denom_i = jnp.sum(inv_denom, axis=-2, keepdims=True) # (1, N)
    B_eff_proj = (1.0 / M) * (alpha * B_eff * sum_inv_denom_i + jnp.swapaxes(A, -2, -1) @ t_inv_denom)

    Z_proj = Z - t * target_penalty

    if residual:
        B_proj = I - B_eff_proj
    else:
        B_proj = B_eff_proj

    return A_proj, B_proj, Z_proj, t

def bilinearProjBatched(x, w, y):
    """
    Project x, w, y such that w @ x = y
    x: (K, batch)
    w: (M, K)
    y: (M, batch)
    y can technically just be a 1D vector if batch is 1, but we assume batched here.
    """
    # We call matmul_proj with A=w, B=x, Z=y
    A_proj, B_proj, Z_proj, _ = matmul_proj(w, x, y, num_steps=10, omega=0.0)
    # Return matched output format: x_proj, w_proj, y_proj
    return B_proj, A_proj, Z_proj

def bilinearMatrixBatched(x, w, y):
    """
    Alternative name if needed based on `projections.bilinearMatrix` usage.
    """
    return bilinearProjBatched(x, w, y)
