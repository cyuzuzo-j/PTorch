
import jax
import jax.numpy as jnp
from pjax.core.ops import haddamarmul_proj
from pjax.core.no_ops import fft2d, ifft2d
def test_fft2d():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (4, 4))
    print("Input x:\n", x)

    x_fft = fft2d(x)
    print("FFT2 of x:\n", x_fft)

    x_ifft = ifft2d(x_fft)
    print(" check if ifft2(fft2(x)) == x:\n", jnp.allclose(x_ifft, x))


def test_haddamarmul():
    key = jax.random.PRNGKey(0)
    key, k1, k2, k3, k4, k5, k6 = jax.random.split(key, 7)
    
    shape = (10,)
    a = jax.random.normal(k1, shape) + 1j * jax.random.normal(k2, shape)
    b = jax.random.normal(k3, shape) + 1j * jax.random.normal(k4, shape)
    y = jax.random.normal(k5, shape) + 1j * jax.random.normal(k6, shape)
    
    print("Input a:", a)
    print("Input b:", b)
    print("Input y:", y)
    
    while True:
        a, b = haddamarmul_proj(a, b, y)
        
        loss = jnp.linalg.norm(a * b - y)
        print( "Loss:", loss)
    
if __name__ == "__main__":
    #test_fft2d()
    test_haddamarmul()
