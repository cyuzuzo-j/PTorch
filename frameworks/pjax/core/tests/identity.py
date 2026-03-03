""""
Test file confirming the functioning of the identity function
"""
from .. import ops
import jax.numpy as jnp
import numpy as np

def test_forward_working():
    a = jnp.array([1])
    np.testing.assert_allclose(ops.identity(a), a)

def test_projection_working():
    a = jnp.array([1])
    z = jnp.array([2])

    analytical_proj = 1.5
    a_proj = ops.identity_proj(a, z)
    np.testing.assert_allclose(a_proj, analytical_proj)
    
