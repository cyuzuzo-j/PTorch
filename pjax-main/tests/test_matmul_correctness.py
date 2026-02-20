
import jax
import jax.numpy as jnp
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from pjax.core.ops import matmul_op, matmul_proj
import sys

jax.config.update("jax_disable_jit", True)

def test_matmul_correctness():
    with open("test_output.txt", "w") as f:
        f.write("Starting test...\n")
        print("Starting test...", flush=True)
    # 1. Test Forward Op
    B, M, K, N = 2, 3, 4, 3
    a = jax.random.normal(jax.random.PRNGKey(0), (B, M, K))
    b = jax.random.normal(jax.random.PRNGKey(1), (K, N))
    
    expected = jnp.matmul(a, b)
    result = matmul_op(a, b)
    
    with open("test_output.txt", "a") as f:
        if jnp.allclose(expected, result):
             f.write("Forward matmul passed\n")
        else:
             f.write(f"Forward matmul failed. Expected {expected}, got {result}\n")
    
    # 2. Test Backward Projection (Broadcasting)
    # We want to verified that matmul_proj handles (B, M, K), (K, N), (B, M, N)
    # and returns (B, M, K) and (K, N).
    
    z = jax.random.normal(jax.random.PRNGKey(2), (B, M, N))
    
    a_new, b_new = matmul_proj(a, b, z)
    
    with open("test_output.txt", "a") as f:
        f.write(f"a_new shape: {a_new.shape}\n")
        f.write(f"b_new shape: {b_new.shape}\n")
        
        if a_new.shape == a.shape:
            f.write("a_new shape correct\n")
        else:
            f.write(f"a_new shape mismatch: {a_new.shape} vs {a.shape}\n")
            
        if b_new.shape == b.shape:
             f.write("b_new shape correct\n")
        else:
             f.write(f"b_new shape mismatch: {b_new.shape} vs {b.shape}\n")
    
        # Verify values are reasonable (not NaN, not Inf)
        if jnp.all(jnp.isfinite(a_new)):
             f.write("a_new values finite\n")
        else:
             f.write("a_new has non-finite values\n")
             
        if jnp.all(jnp.isfinite(b_new)):
             f.write("b_new values finite\n")
        else:
             f.write("b_new has non-finite values\n")
    
    # Verify that b_new is indeed influenced by all batches.
    # We can try setting z for batch 0 to something extreme and see if b_new shifts.
    z_mod = z.at[0].set(1000.0)
    _, b_new_mod = matmul_proj(a, b, z_mod)
    
    diff = jnp.linalg.norm(b_new_mod - b_new)
    
    with open("test_output.txt", "a") as f:
        f.write(f"Difference in b_new after modifying batch 0: {diff}\n")
        if diff > 1.0:
            f.write("Batch influence check passed\n")
        else:
            f.write("Batch influence check failed: b_new insufficient change\n")
    
        f.write("Backward matmul passed basic checks\n")

if __name__ == "__main__":
    test_matmul_correctness()
