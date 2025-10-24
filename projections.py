import jax
import jax.numpy as jnp

@jax.jit
def bilinearProj(a, b, z, steps = 10):
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


def bilinearMatrix(a, B, z, steps = 10):
    """Project onto bilinear function graph using Newton's method."""
    """ with A a vector of shape (dim,), B a matrix of shape (AA, dim) and z a vector of shape (AA,)"""
    a_buffer = jax.numpy.zeros((B.shape[0], a.shape[0]))
    projection_B = jax.numpy.zeros_like(B)    
    def proj_single(b, zi):
        return bilinearProj(a, b, zi, steps)
    a_buffer, projection_B, _ = jax.vmap(proj_single)(B, z)
    proj_a = jax.numpy.mean(a_buffer, axis=0)
    return proj_a, projection_B, z

def bilinearMatrixs(X, W, Z, steps = 10):
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
            #print("before",x.T @ w - zi)
            x_buffer, w_buffer, _ = bilinearProj(x, w, zi, steps)
            #print("after",x_buffer.T @ w_buffer - zi, jnp.linalg.norm(x_buffer - x), jnp.linalg.norm(w_buffer - w))
            W_buffer = W_buffer.at[sample,output,:].set(w_buffer)
            X_buffer = X_buffer.at[output,:,sample].set(x_buffer)
    proj_X = jax.numpy.mean(X_buffer, axis=0)
    #print( jnp.linalg.norm(proj_X - X))
    proj_W = jax.numpy.mean(W_buffer, axis=0)
    #print(jnp.linalg.norm(proj_W - W))
    return proj_X, proj_W, Z


def bilinearMatrixsJAX(X, W, Z, steps=10):
    """Project onto bilinear function graph using Newton's method (vectorized)."""

    # Function to apply bilinearProj to one (x, z) pair for all outputs
    def process_sample(x, z):
        # Vectorize over outputs (axis 0 of W and Z)
        def process_output(w, zi):
            x_buffer, w_buffer, _ = bilinearProj(x, w, zi, steps)
            return x_buffer, w_buffer

        x_bufs, w_bufs = jax.vmap(process_output, in_axes=(0, 0))(W, z)
        return x_bufs, w_bufs

    # Vectorize over samples (axis 1 of X and Z)
    X_buffers, W_buffers = jax.vmap(process_sample, in_axes=(1, 1))(X, Z)

    # X_buffers shape: (num_samples, num_outputs, dim)
    # W_buffers shape: (num_samples, num_outputs, dim_w)

    # Average over samples
    proj_X = jnp.mean(jnp.swapaxes(X_buffers, 0, 1), axis=0)
    print(proj_X.shape)
    proj_W = jnp.mean(W_buffers, axis=0)
    
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

def stepActivation(x, W, y):
    """Project onto step activation function constraint: y = step(x) where step(x) = 1 if x >= 0, 0 otherwise."""
    y_projected = jnp.zeros_like(y)
    for i in range(W.shape[0]):
        w = W[i,:]
        h = w@x
        y_projected.at[i].set(jax.lax.cond(h>= 0, lambda: 1.0, lambda: 0.0))
    return x, W, y_projected



"""----------- Bilinear projection with only a and b updated from pjax"""
""" https://arxiv.org/pdf/2206.04878 komt van hier, lichtelijk anders dan die van learning without loss"""
@jax.jit
def bilinear_proj_orr(a, b, z, steps = 10):
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

def bilinearMatrix_orr(a, B, z, steps = 10):
    """Project onto bilinear function graph using Newton's method."""
    """ with A a vector of shape (dim,), B a matrix of shape (AA, dim) and z a vector of shape (AA,)"""
    a_buffer = jax.numpy.zeros((B.shape[0], a.shape[0]))
    projection_B = jax.numpy.zeros_like(B)    
    def proj_single(b, zi):
        return bilinear_proj_orr(a, b, zi, steps)
    a_buffer, projection_B = jax.vmap(proj_single)(B, z)
    proj_a = jax.numpy.mean(a_buffer, axis=0)
    return proj_a, projection_B, z

