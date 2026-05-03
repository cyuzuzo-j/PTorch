import jax
import jax.numpy as jnp
import jax.lax as lax

def newton_schulz_orthogonalize_non_square(A, num_iters=2):
    """
    Orthogonalize the columns of A (m x n, m >= n) using Newton-Schulz iteration.
    Returns Q with orthonormal columns: Q^T Q ≈ I
    """
    M = A.T @ A  # n x n
    X = jnp.eye(M.shape[0])  # initial guess
    
    for _ in range(num_iters):
        X = 0.5 * X @ (3 * jnp.eye(M.shape[0]) - M @ X @ X)
    
    Q = A @ X
    return Q

def orthonormalizeMatrixNS(x,W,y):
    Q = newton_schulz_orthogonalize_non_square(W, num_iters=10)
    return x, Q, y

def orthonormalizeMatrix(x,W,y):
    # SVD
    U, S, Vt = jnp.linalg.svd(W, full_matrices=False)

    # Closest orthonormal matrix Q
    Q = U @ Vt
    return x, Q, y

def solve_lambda(W0, X0, y0):
    """
    """
    return 0.5*(1-jnp.sqrt((jnp.linalg.norm(W0 - X0)**2 )/ y0))

def minusProjectionThing(x, w, y, steps=1):
    """Use jax.lax.cond instead of if statements for traced arrays."""
    lambda1 = solve_lambda(w, x, y)
    if jnp.isnan(lambda1):
        print("NaN case encountered in minusProjectionThing")
        return x, w, y
    else:
        denom = 1 - 2 * lambda1
        
        # Check for numerical instability
        if jnp.abs(denom) < 1e-8:
            print(f"Warning: Numerical instability detected. Denominator (1 - 2*lambda1) = {denom} is close to zero.")
        
        alpha = (1 - lambda1) / denom
        beta = lambda1 / denom  
        x_new = x * alpha - w * beta
        w_new = w * alpha - x * beta
        return x_new, w_new, y

@jax.jit
def minusThingMatrix(a, B, z, steps=10):
    a_buffer = jax.numpy.zeros((B.shape[0], a.shape[0]))
    projection_B = jax.numpy.zeros_like(B)  
      
    def proj_single(b, zi):
        return minusProjectionThing(a, b, zi)
    a_buffer, projection_B, proj_z = jax.vmap(proj_single)(B, z)
    proj_a = jax.numpy.mean(a_buffer, axis=0)
    return proj_a, projection_B, proj_z

@jax.jit
def bilinearProj(a, b, z, steps=10):
    """Project onto bilinear function graph using Newton's method."""
    p = a @ b
    q = a @ a + b @ b
    def f(t):
        return ((1 + t**2) * p + t * q) / (1 - t**2) ** 2 - z 

    f_prime = jax.grad(f)

    def newton_step(t):
        return t - f(t) / f_prime(t)

    t = jax.lax.fori_loop(0, steps, lambda _, t: newton_step(t), 0.0, unroll=False)

    a_new = (a + t * b) / (1 - t**2)
    b_new = (b + t * a) / (1 - t**2)

    return a_new, b_new, z

@jax.jit
def bilinearMatrix(a, B, z, steps = 20):
    """Project onto bilinear function graph using Newton's method."""
    """ with A a vector of shape (dim,), B a matrix of shape (AA, dim) and z a vector of shape (AA,)"""
    a_buffer = jax.numpy.zeros((B.shape[0], a.shape[0]))
    projection_B = jax.numpy.zeros_like(B)    
    def proj_single(b, zi):
        return bilinearProj(a, b, zi, steps)
    a_buffer, projection_B, _ = jax.vmap(proj_single)(B, z)
    proj_a = jax.numpy.mean(a_buffer, axis=0)
    return proj_a, projection_B, z

@jax.jit
def matmul_proj(X, W, Z, alpha=1.0, g=1.0, omega=1.0, num_steps=10, residual=False):
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
def stepActivationVectorized(x, W, y):
    """
    Vectorized step activation projection over batch.
    y = step(x). 1 if x >= 0 else -1.
    x is (dim, batch). W is (dim_out, dim_in).
    """
    H = W @ x
    y_projected = jax.numpy.where(H >= 0, 1.0, -1.0)
    return x, W, y_projected


@jax.jit
def reluActivationVectorized(x, W, z):
    """
    Vectorized ReLU activation projection.
    x is the pre-activation input, W is the identity matrix, z is the target post-activation.
    """
    H = W @ x

    x_1 = jnp.clip(H, a_max=0)
    dist_1 = (H - x_1) ** 2 + z ** 2

    x_2 = jnp.clip((H + z) / 2.0, a_min=0)
    dist_2 = (H - x_2) ** 2 + (z - x_2) ** 2

    new_H = jnp.where(dist_1 < dist_2, x_1, x_2)
    new_z = jnp.where(dist_1 < dist_2, 0.0, x_2)

    return new_H, W, new_z


@jax.jit
def leakyReluActivationVectorized(x, W, z, slope=0.1):
    """
    Vectorized Leaky ReLU activation projection.
    x is the pre-activation input, W is the identity matrix, z is the target post-activation.
    """
    H = W @ x

    x_1 = jnp.clip((H + slope * z) / (1.0 + slope * slope), a_max=0)
    y_1 = slope * x_1
    dist_1 = (H - x_1) ** 2 + (z - y_1) ** 2

    x_2 = jnp.clip((H + z) / 2.0, a_min=0)
    y_2 = x_2
    dist_2 = (H - x_2) ** 2 + (z - y_2) ** 2

    new_H = jnp.where(dist_1 < dist_2, x_1, x_2)
    new_z = jnp.where(dist_1 < dist_2, y_1, y_2)

    return new_H, W, new_z

def pos(x, w, y):
    """Project onto positive orthant."""
    return jax.numpy.maximum(x, 0), jax.numpy.maximum(w, 0), y

def normalizeWeights(x, w, y, norm=1):
    """ Project onto weight norm constraint"""
    w_norm = jax.numpy.linalg.norm(w)
    w = jax.lax.cond(w_norm > norm, lambda w: w * (norm / w_norm), lambda w: w, w)
    return x, w, y

## helper functions for classifier output
def _largerThenDelta(x,delta):
    return jax.numpy.maximum(x, delta)

def _smallerThenZero(x,delta):
    return jax.numpy.minimum(x,0)


def classifierOutput(x,w,y, delta=1):
    """ project onto classification output constraints y >= delta for positive class, y <= 0 for negative class"""
    x = jax.lax.cond(y == 1, _largerThenDelta, _smallerThenZero, x,delta)
    return x,w,y
@jax.jit
def classifierOutputVector(x, w, y, delta=1):
    """ project onto classification output constraints y >= delta for positive class, y <= 0 for negative class"""
    x_new = jnp.where(y == 1, jnp.maximum(x, delta), jnp.minimum(x, 0))
    return x_new, w, y

@jax.jit
def stepActivation(x, W, y):
    """Project onto step activation function constraint: y = step(x) where step(x) = 1 if x >= 0, 0 otherwise."""
    y_projected = jnp.zeros_like(y)
    for i in range(W.shape[0]):
        w = W[i,:]
        h = w@x
        y_projected.at[i].set(jax.lax.cond(h>= 0, lambda: 1, lambda:-1))
    return x, W, y_projected

def sum_relu_proj(x,W, y):
    """Project onto sum-ReLU function graph."""
    inputs = x
    outputs = y
    new_inputs = jnp.zeros_like(inputs)
    new_outputs = jnp.zeros_like(outputs)
    for i, (input, output ) in enumerate(zip(inputs, outputs)):
        new_value = (input + output)/2
        new_input, new_output = new_value, new_value
        new_output2 = 0
        new_input2 = input
        
        dist1 = (jnp.abs(input - new_input))**2 + (jnp.abs(output - new_output))**2
        dist2 = output**2
        new_inputs = new_inputs.at[i].set(jnp.where(dist1 < dist2, new_input, new_input2))
        new_outputs = new_outputs.at[i].set(jnp.where(dist1 < dist2, new_output, new_output2))
    return new_inputs , W, new_outputs

@jax.jit
def matmul_proj_fixed_X(X, W, Z, g=1.0, omega=1.0):
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
def hyperbolaProj(u0, v0, gamma, num_steps=20, eps=1e-4):
    """
    Project (u0, v0) onto the hyperbola C_gamma = {(u,v) : ||u||^2 - ||v||^2 = 2*gamma}
    in a Hilbert space, for any sign of gamma.

    By Prop 3.3 of Bauschke-Lal-Wang, the optimum has u = alpha * u0, v = beta * v0
    with alpha, beta >= 0. KKT gives
        alpha = 1/(1-lam),  beta = 1/(1+lam),  lam in (-1, 1),
    where lam solves the scalar equation
        g(lam) := a/(1-lam)^2 - b/(1+lam)^2 = 2*gamma,
    with a = ||u0||^2, b = ||v0||^2. g is strictly increasing on (-1,1) and ranges over
    R, so a unique solution exists for any gamma in R.

    Theorem 5.1 (gamma < 0) is automatic here: the swap symmetry
        P_{C_gamma}(u0, v0) = swap(P_{C_{-gamma}}(v0, u0))
    corresponds to lam -> -lam in (*); the same Newton solver handles both cases.
    """
    a = jnp.sum(u0 * u0)
    b = jnp.sum(v0 * v0)
    target = 2.0 * gamma
    bound = 1.0 - eps

    def step(_, lam):
        one_minus = 1.0 - lam
        one_plus  = 1.0 + lam
        g_val   = a / (one_minus ** 2) - b / (one_plus ** 2) - target
        g_prime = 2.0 * a / (one_minus ** 3) + 2.0 * b / (one_plus ** 3)
        lam_new = lam - g_val / (g_prime + 1e-12)
        return jnp.clip(lam_new, -bound, bound)

    lam = jax.lax.fori_loop(0, num_steps, step, 0.0)
    alpha = 1.0 / (1.0 - lam)
    beta  = 1.0 / (1.0 + lam)
    return alpha * u0, beta * v0


@jax.jit
def bilinearProjViaHyperbola(a, b, z, num_steps=20):
    """
    Project (a, b) onto {(a',b') : <a',b'> = z} by routing through the hyperbola
    formulation: rotate by pi/4 to (u, v) with ||u||^2 - ||v||^2 = 2z, project with
    `hyperbolaProj`, then rotate back. Equivalent to `bilinearProj` but expressed in
    the hyperbola variables; supports any sign of z including z < 0 via Thm 5.1.
    """
    inv_sqrt2 = 1.0 / jnp.sqrt(2.0)
    u0 = (a + b) * inv_sqrt2
    v0 = (b - a) * inv_sqrt2
    u, v = hyperbolaProj(u0, v0, z, num_steps)
    a_new = (u - v) * inv_sqrt2
    b_new = (u + v) * inv_sqrt2
    return a_new, b_new, z


@jax.jit
def matmul_proj_hyperbola(X, W, Z, num_steps=20, eps=1e-4):
    """
    Hyperbola-form variant of `matmul_proj`: projects onto the per-pair / per-row
    bilinear constraint W[j,:] . X[:,b] = Z[j,b] in isolation, then takes the consensus
    average across the shared dimensions (b for rows of W, j for columns of X).

    Uses the Newton equation
        a_{jb}/(1-lam)^2 - b_{jb}/(1+lam)^2 = 2 * Z[j,b]
    with
        a_{jb} = (||w_j||^2 + ||x_b||^2)/2 + <w_j, x_b>,
        b_{jb} = (||w_j||^2 + ||x_b||^2)/2 - <w_j, x_b>,
    derived from the pi/4 rotation. The reconstruction is the standard
        x' = (x + lam y)/(1-lam^2),  y' = (lam x + y)/(1-lam^2),
    which fuses with the consensus averaging into two 2D matmuls (no MNK tensor).

    X: (dim_in, N), W: (M, dim_in), Z: (M, N).
    """
    A = jnp.atleast_2d(W)         # (M, K)
    B = jnp.atleast_2d(X)         # (K, N)
    Z_target = jnp.atleast_2d(Z)  # (M, N)

    M, K = A.shape
    N = B.shape[1]

    p  = A @ B                                                # (M, N)  <w_j, x_b>
    qa = jnp.sum(A * A, axis=-1, keepdims=True)               # (M, 1)
    qb = jnp.sum(B * B, axis=-2, keepdims=True)               # (1, N)
    half_q = 0.5 * (qa + qb)                                  # (M, N)
    a_pair = half_q + p                                       # ||u||^2
    b_pair = half_q - p                                       # ||v||^2

    bound = 1.0 - eps

    def newton_step(_, lam):
        one_minus = 1.0 - lam
        one_plus  = 1.0 + lam
        g_val   = a_pair / (one_minus ** 2) - b_pair / (one_plus ** 2) - 2.0 * Z_target
        g_prime = 2.0 * a_pair / (one_minus ** 3) + 2.0 * b_pair / (one_plus ** 3)
        lam_new = lam - g_val / (g_prime + 1e-12)
        return jnp.clip(lam_new, -bound, bound)

    lam = jax.lax.fori_loop(0, num_steps, newton_step, jnp.zeros_like(p))
    lam = jnp.clip(lam, -bound, bound)

    inv_denom    = 1.0 / (1.0 - lam ** 2)                     # (M, N)
    lam_inv_denom = lam * inv_denom                           # (M, N)

    # Per-pair: w_j' = (w_j + lam x_b)/(1-lam^2), x_b' = (lam w_j + x_b)/(1-lam^2).
    # Consensus on rows of W (avg over b=1..N) and cols of X (avg over j=1..M).
    sum_inv_j = jnp.sum(inv_denom, axis=-1, keepdims=True)    # (M, 1)
    A_proj = (1.0 / N) * (A * sum_inv_j + lam_inv_denom @ B.T)

    sum_inv_i = jnp.sum(inv_denom, axis=-2, keepdims=True)    # (1, N)
    B_proj = (1.0 / M) * (B * sum_inv_i + A.T @ lam_inv_denom)

    return B_proj.reshape(X.shape), A_proj.reshape(W.shape), Z_target.reshape(Z.shape)


@jax.jit
def matmul_proj_fixed_X_hyperbola(X, W, Z, num_steps=20, eps=1e-4):
    """
    X-fixed variant of `matmul_proj_hyperbola`: averages only across N (no consensus on X).
    Use when the input activations are clamped (e.g. x_in = batch data).
    """
    A = jnp.atleast_2d(W)
    B = jnp.atleast_2d(X)
    Z_target = jnp.atleast_2d(Z)

    M, K = A.shape
    N = B.shape[1]

    p  = A @ B
    qa = jnp.sum(A * A, axis=-1, keepdims=True)
    qb = jnp.sum(B * B, axis=-2, keepdims=True)
    half_q = 0.5 * (qa + qb)
    a_pair = half_q + p
    b_pair = half_q - p

    bound = 1.0 - eps

    def newton_step(_, lam):
        one_minus = 1.0 - lam
        one_plus  = 1.0 + lam
        g_val   = a_pair / (one_minus ** 2) - b_pair / (one_plus ** 2) - 2.0 * Z_target
        g_prime = 2.0 * a_pair / (one_minus ** 3) + 2.0 * b_pair / (one_plus ** 3)
        lam_new = lam - g_val / (g_prime + 1e-12)
        return jnp.clip(lam_new, -bound, bound)

    lam = jax.lax.fori_loop(0, num_steps, newton_step, jnp.zeros_like(p))
    lam = jnp.clip(lam, -bound, bound)

    inv_denom    = 1.0 / (1.0 - lam ** 2)
    lam_inv_denom = lam * inv_denom

    sum_inv_j = jnp.sum(inv_denom, axis=-1, keepdims=True)
    A_proj = (1.0 / N) * (A * sum_inv_j + lam_inv_denom @ B.T)

    # Z reconstructed from the projected (W, X=fixed): w_j' . x_b - tracked via lam.
    Z_proj = (a_pair / (1.0 - lam) ** 2 - b_pair / (1.0 + lam) ** 2) * 0.5

    return X.reshape(X.shape), A_proj.reshape(W.shape), Z_proj.reshape(Z.shape)


@jax.jit
def bilinear_proj_orr(a, b, z, steps=10):
    """Project onto bilinear function graph using Newton's method."""
    p = a @ b
    q = a @ a + b @ b

    def f(t):
        return ((1 + t**2) * p + t * q) / (1 - t**2) ** 2 - z + t

    f_prime = jax.grad(f)

    def newton_step(t):
        return t - f(t) / f_prime(t)

    t = jax.lax.fori_loop(0, steps, lambda _, t: newton_step(t), 0.0, unroll=True)

    a_new = (a + t * b) / (1 - t**2)
    b_new = (b + t * a) / (1 - t**2)

    return a_new, b_new

def bilinearMatrix_orr(a, B, z, steps=10):
    """Project onto bilinear function graph using Newton's method."""
    """ with A a vector of shape (dim,), B a matrix of shape (AA, dim) and z a vector of shape (AA,)"""
    a_buffer = jax.numpy.zeros((B.shape[0], a.shape[0]))
    projection_B = jax.numpy.zeros_like(B)    
    def proj_single(b, zi):
        return bilinear_proj_orr(a, b, zi, steps)
    a_buffer, projection_B = jax.vmap(proj_single)(B, z)
    proj_a = jax.numpy.mean(a_buffer, axis=0)
    return proj_a, projection_B, z

