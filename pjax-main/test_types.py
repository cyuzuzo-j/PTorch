
import jax
import jax.numpy as jnp

def test_types():
    x = jnp.array([1.0+1.0j], dtype=jnp.complex64)
    print(f"Input type: {x.dtype}")
    
    # Simulating reflection
    res = 2.0 * x - x
    print(f"Reflection (2.0 * x - x) type: {res.dtype}")
    
    # Simulating AP
    res_ap = x
    print(f"AP type: {res_ap.dtype}")
    
    # Check scan compatibility
    def step(carry, _):
        val = carry
        new_val = 2.0 * val - val
        return new_val, None
        
    try:
        final, _ = jax.lax.scan(step, x, None, length=1)
        print(f"Scan result type: {final.dtype}")
    except Exception as e:
        print(f"Scan failed: {e}")

if __name__ == "__main__":
    test_types()
