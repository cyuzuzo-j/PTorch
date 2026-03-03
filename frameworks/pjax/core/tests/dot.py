""""
Test file confirming the functioning of the dot product 

"""
from .. import ops
import jax.numpy as jnp
from scipy.optimize import minimize
import numpy as np

def test_forward_working():
    a = jnp.array([1, 2])
    b = jnp.array([4, 5])
    assert ops.dot(a, b) == jnp.dot(a, b)

def test_projection_trivial():
    # trivial projection
    a = jnp.array([0.0, 0.0])
    b = jnp.array([0.0, 0.0])
    z = jnp.array(0.0)
    a_proj, b_proj = ops.bilinear_proj(a, b, z)
    np.testing.assert_allclose(a_proj, a)
    np.testing.assert_allclose(b_proj, b)

def test_projection_non_trivial():
    ## given a new target z, we project a and b onto the graph of the dot product
    ## we compare the result with scipy's minimize function
    a = jnp.array([1.0, 2.0])
    b = jnp.array([3.0, 4.0])
    z = jnp.array(5.0)

    
    a_np = np.array(a)
    b_np = np.array(b)
    z_np = float(z)
    
    def objective(x):
        a_curr = x[:len(a_np)]
        b_curr = x[len(a_np):]
        return np.sum((a_curr - a_np)**2) + np.sum((b_curr - b_np)**2)
        
    def constraint(x):
        a_curr = x[:len(a_np)]
        b_curr = x[len(a_np):]
        return np.dot(a_curr, b_curr) - z_np

    x0 = np.concatenate([a_np, b_np], dtype=np.float64)
    res = minimize(
        objective, 
        x0, 
        constraints={'type': 'eq', 'fun': constraint}
    )
    
    a_scipy = res.x[:len(a_np)]
    b_scipy = res.x[len(a_np):]
    
    a_pjax, b_pjax = ops.bilinear_proj(a, b, z)
    
    np.testing.assert_allclose(a_pjax, a_scipy, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(b_pjax, b_scipy, rtol=1e-4, atol=1e-4)

def test_projection_idempotent():
    # projection of projection is the same
    a = jnp.array([1.0, 2.0])
    b = jnp.array([3.0, 4.0])
    z = jnp.array(5.0)
    a_pjax, b_pjax = ops.bilinear_proj(a, b, z)
    a_pjax2, b_pjax2 = ops.bilinear_proj(a_pjax, b_pjax, z)
    np.testing.assert_allclose(a_pjax2, a_pjax, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(b_pjax2, b_pjax, rtol=1e-4, atol=1e-4)