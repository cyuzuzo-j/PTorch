import jax

@jax.jit
def bilinearProj(a, b, z, steps = 10):
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

def bilinearMatrix(a, B, z, steps = 10):
    """Project onto bilinear function graph using Newton's method."""
    """ with A a vector of shape (dim,), B a matrix of shape (AA, dim) and z a vector of shape (AA,)"""
    a_buffer = jax.numpy.zeros((B.shape[0], a.shape[0]))
    projection_B = jax.numpy.zeros_like(B)
    for i in range(B.shape[0]):
        b = B[i,:]
        temp_proj_a, proj_b, _ = bilinearProj(a, b, z[i], steps)
        a_buffer = a_buffer.at[i, :].set(temp_proj_a)
        projection_B = projection_B.at[i,:].set(proj_b)
    proj_a = jax.numpy.mean(a_buffer, axis=0)
    return proj_a, projection_B, z
        
        
def pos(x, w, y):
    """Project onto positive orthant."""
    return jax.numpy.maximum(x, 0), jax.numpy.maximum(w, 0), y

def classifierOutput(x,w,y, delta=1):
    """ project onto classification output constraints y >= delta for positive class, y <= 0 for negative class"""
    if y == 1:
        x = jax.numpy.maximum(x, delta)
    else:
        x = jax.numpy.minimum(x, 0)
    return x,w,y
