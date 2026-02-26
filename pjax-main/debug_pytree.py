import jax
import sys
from pjax.core.computation import Parameter, Operation, array
import jax.numpy as jnp

sys.setrecursionlimit(500)

print("Starting debug...")
param = Parameter(jnp.array([1.0]), name="P1")
print("Created param")
op = Operation("dummy_op", lambda x: x, lambda x, out: x, [param], jnp.array([1.0]))
print("Created op")

def my_map(x):
    print("MAPPING OVER", type(x))
    return x

try:
    print("Flattening op...")
    flat, treedef = jax.tree_util.tree_flatten(op)
    print("Flattened length:", len(flat))
    print("Tredef:", treedef)
except Exception as e:
    print("EXCEPTION:", e)

try:
    print("Calling tree.map...")
    jax.tree.map(my_map, op)
    print("Done tree map")
except Exception as e:
    print("EXCEPTION:", e)
