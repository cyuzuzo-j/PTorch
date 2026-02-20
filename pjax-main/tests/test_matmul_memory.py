
import jax
import jax.numpy as jnp
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from pjax.core.api import matmul
from pjax.core.computation import array
from pjax.optim_efficient import get_graph
from pjax.core.no_ops import repeat

def test_matmul_graph_efficiency():
    # Setup
    B, M, K, N = 10, 32, 64, 32
    a_shape = (B, M, K)
    b_shape = (K, N)
    
    a_val = jnp.zeros(a_shape)
    b_val = jnp.zeros(b_shape)
    
    a = array(a_val, name="A")
    b = array(b_val, name="B")
    
    # Run matmul
    print("Running matmul...")
    out = matmul(a, b)
    
    # Analyze Graph
    graph = get_graph(out)
    
    repeat_nodes = [node for node in graph.nodes if getattr(node, 'name', '').startswith('repeat') or 'repeat' in str(type(node))]
    # Or specifically check for Repeat transform usage via no_ops.repeat
    # In api.py, repeat is called directly. It creates ShapeTransform nodes.
    # We can check the transform function associated with ShapeTransforms.
    
    shape_transforms = [node for node in graph.nodes if hasattr(node, 'transform')]
    repeats_found = 0
    for node in shape_transforms:
        if node.transform == repeat.transform: 
             repeats_found += 1
             print(f"Found explicit repeat node: {node.name}")

    print(f"Total Repeat nodes found: {repeats_found}")
    
    # Expectation: Current implementation has explicit repeats for broadcasting B
    if repeats_found > 0:
        print("FAIL: Explicit Repeats present (Baseline validated).")
    else:
        print("SUCCESS: No explicit Repeats found.")

if __name__ == "__main__":
    test_matmul_graph_efficiency()
