
import jax
import jax.numpy as jnp
import pjax
from pjax import nn, optim
import numpy as np

def test_dot_complex():
    # Create complex arrays
    a = jnp.array([1.0 + 1.0j, 2.0 + 2.0j])
    b = jnp.array([1.0 - 1.0j, 2.0 - 2.0j])
    
    # Expected dot product: (1+i)(1-i) + (2+2i)(2-2i) = (1+1) + (4+4) = 2 + 8 = 10
    
    # Use pjax.dot
    # We need to wrap it in a function to test projection?
    # Or just run it.
    
    res = pjax.dot(a, b)
    print(f"Dot result: {res}")
    
    # Now test projection (backward pass equivalent in pjax?)
    # pjax.dot is a Computation.
    # We can use optim.AlternatingProjections to see if it updates correctly?
    
    # Let's just check if the projection function casts to float.
    # We can access the projection function from the primitive?
    # pjax.core.ops.dot is the primitive.
    
    from pjax.core.ops import bilinear_proj
    
    # dot.projection is the projection function
    # It takes (inputs..., output)
    
    a_in = jnp.array([1.0 + 1.0j])
    b_in = jnp.array([1.0 + 1.0j])
    z_out = jnp.array(2.0j) # Target
    
    # Project a, b to match a*b = 2j
    # If it casts to float, it will ignore imaginary parts.
    
    proj_a, proj_b = bilinear_proj(a_in, b_in, z_out)
    
    print(f"Input a: {a_in}, b: {b_in}")
    print(f"Target z: {z_out}")
    print(f"Projected a: {proj_a}")
    print(f"Projected b: {proj_b}")
    print(f"Projected product: {proj_a * proj_b}")
    
    if jnp.iscomplexobj(proj_a):
        print("Projected a is complex.")
    else:
        print("Projected a is NOT complex.")

if __name__ == "__main__":
    test_dot_complex()
