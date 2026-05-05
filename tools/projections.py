import jax
import jax.numpy as jnp
import jax.lax as lax


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
