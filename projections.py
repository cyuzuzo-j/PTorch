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
        return ((1 + t**2) * p + t * q) / (1 - t**2) ** 2 - z + t
 
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

def bilinearMatrixs(X, W, Z, steps=10):
    """Project onto bilinear function graph using Newton's method."""
    """ with A a vector of shape (dim,), B a matrix of shape (AA, dim) and z a vector of shape (AA,)"""
    X = jax.numpy.array(X)
    W = jax.numpy.array(W)
    
    W_buffer = jax.numpy.zeros((X.shape[1], W.shape[0], W.shape[1]))
    X_buffer = jax.numpy.zeros((W.shape[0], X.shape[0], X.shape[1]))
    for sample in range(X.shape[1]):
        x = X[:,sample]
        z = Z[:,sample]
        for output in range(W.shape[0]):
            w = W[output,:]
            zi = z[output]
            x_buffer, w_buffer, _ = bilinearProj(x, w, zi, steps)
            W_buffer = W_buffer.at[sample,output,:].set(w_buffer)
            X_buffer = X_buffer.at[output,:,sample].set(x_buffer)
    proj_X = jax.numpy.mean(X_buffer, axis=0)
    proj_W = jax.numpy.mean(W_buffer, axis=0)
    return proj_X, proj_W, Z


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
def classifierOutputVector(x,w,y, delta=1):
    """ project onto classification output constraints y >= delta for positive class, y <= 0 for negative class"""
    x_new= jnp.zeros_like(y)
    for k in range(y.shape[0]):
        x_new = x_new.at[k].set(jax.lax.cond(y[k] == 1, _largerThenDelta, _smallerThenZero, x[k],delta))
    return x_new,w,y

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

