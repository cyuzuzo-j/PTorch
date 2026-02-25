import jax
from pjax.core.computation import Parameter, Operation, array
import jax.numpy as jnp

param = Parameter(jnp.array([1.0]))
op = Operation("dummy", lambda x: x, lambda x, out: x, [param], jnp.array([1.0]))

def my_map(x):
    print("MAPPING OVER", x)
    return x

jax.tree.map(my_map, op)
