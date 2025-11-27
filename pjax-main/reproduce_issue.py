
import jax
import jax.numpy as jnp
from pjax.core.ops import haddamarmul_proj

def test_haddamarmul():
    # From aaa.py: const = jnp.array([1.0, 2.0, -1.0, 0.5, 0.2, -0.4])
    # w_r, w_i, x_r, x_i, y_r, y_i
    
    w_r = 1.0
    w_i = 9
    x_r = -1.0
    x_i = 0.5
    y_r = 0.2
    y_i = -0.4
    
    a = jnp.array([w_r + 1j * w_i, 2.0 + 3.0j])
    b = jnp.array([x_r + 1j * x_i, 1.0])
    y = jnp.array([y_r + 1j * y_i, 2.0 + 3.0j])
    
    print("Input a:", a)
    print("Input b:", b)
    print("Input y:", y)
    
    while True:
        a, b = haddamarmul_proj(a, b, y)
        
        print( "new a*b:", a * b)
        print( "Expected y:", y)
    
if __name__ == "__main__":
    test_haddamarmul()
