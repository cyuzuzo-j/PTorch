""""
Test file confirming the functioning of the maxpool function
"""
from .. import ops
import jax.numpy as jnp
from scipy.optimize import minimize
import numpy as np

def test_forward_working():
    a = jnp.array([1, 2, 3])
    np.testing.assert_allclose(ops.maxpool_op(a), 3.0)
