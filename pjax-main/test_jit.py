import jax
import jax.numpy as jnp
from pjax.optim_static import AlternatingProjections
from pjax.core.computation import Parameter, Operation, Computation

def dummy_fun(params):
    # Just a dummy graph
    p = params['w']
    # A fake operation for testing
    from pjax.nn import Linear
    return p

opt = AlternatingProjections()

@jax.jit
def test_fn(x):
    return x * 2

print("JAX works.")
