""""
Test file confirming the functioning of the sum function
"""
from .. import ops
import jax.numpy as jnp
import numpy as np

def test_forward_working():
    a = jnp.array([1, 2, 3])
    np.testing.assert_allclose(ops.sum_op(a), 6.0)

def test_projection_working():
    a = jnp.array([1, 2, 3])
    z = jnp.array(2.0)

    analytical_proj = jnp.array([1-1, 2-1, 3-1]    )
    a_proj = ops.sum_proj(a, z)[0]
    np.testing.assert_allclose(a_proj, analytical_proj)

def test_projection_idempotent():
    a = jnp.array([1, 2, 3])
    z = jnp.array(2.0)
    a_proj = ops.sum_proj(a, z)[0]
    z_proj = jnp.array(2.0 +1)
    a_proj_proj = ops.sum_proj(a_proj, z_proj)[0]
    np.testing.assert_allclose(a_proj, a_proj_proj)  
    
