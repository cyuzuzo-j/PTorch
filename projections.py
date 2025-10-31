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


@jax.jit
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
        # solution 1
        new_value = (input + output)/2
        new_input, new_output = new_value, new_value

        # solution 2
        new_output2 = 0
        new_input2 = input
        
        dist1 = (jnp.abs(input - new_input))**2 + (jnp.abs(output - new_output))**2
        dist2 = output**2
        new_inputs = new_inputs.at[i].set(jnp.where(dist1 < dist2, new_input, new_input2))
        new_outputs = new_outputs.at[i].set(jnp.where(dist1 < dist2, new_output, new_output2))
        

    # select solution minimizing the distance
    return new_inputs , W, new_outputs


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

