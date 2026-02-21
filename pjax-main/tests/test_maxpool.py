
import jax
import jax.numpy as jnp
import numpy as np
from pjax.nn import MaxPool2D
import pjax
from pjax.core import ops

def test_maxpool2d_valid():
    print("Testing MaxPool2D with VALID padding...")
    # Input: (Batch, Height, Width, Channels)
    # Shape: (1, 4, 4, 1)
    input_data = jnp.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ], dtype=jnp.float32).reshape(1, 4, 4, 1)

    # MaxPool: pool_size=(2, 2), strides=(2, 2), padding="VALID"
    max_pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="VALID")
    
    output = max_pool(input_data)
    
    expected_output = jnp.array([
        [6, 8],
        [14, 16]
    ], dtype=jnp.float32).reshape(1, 2, 2, 1)
    
    assert output.shape == expected_output.shape, f"Expected shape {expected_output.shape}, got {output.shape}"
    np.testing.assert_array_equal(output, expected_output)
    print("MaxPool2D VALID padding test passed!")

def test_maxpool2d_same():
    print("Testing MaxPool2D with SAME padding...")
    # Input: (Batch, Height, Width, Channels)
    # Shape: (1, 3, 3, 1)
    input_data = jnp.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=jnp.float32).reshape(1, 3, 3, 1)

    # MaxPool: pool_size=(2, 2), strides=(2, 2), padding="SAME"
    # With SAME padding, it should pad to allow covering the last row/col
    max_pool = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")
    
    output = max_pool(input_data)
    
    # Expected output shape: ceil(3/2) = 2 -> (1, 2, 2, 1)
    # Windows:
    # [1, 2], [4, 5] -> max 5
    # [3, 0], [6, 0] -> max 6 (padded with 0 or -inf? usually pjax.conv_patch handles padding)
    # [7, 8], [0, 0] -> max 8
    # [9, 0], [0, 0] -> max 9
    
    # Note: pjax.conv_patch implementation details matter here. 
    # Assuming standard behavior where padding doesn't introduce values larger than input if input > padding_value.
    
    expected_output = jnp.array([
        [5, 6],
        [8, 9]
    ], dtype=jnp.float32).reshape(1, 2, 2, 1)

    assert output.shape == expected_output.shape, f"Expected shape {expected_output.shape}, got {output.shape}"
    np.testing.assert_array_equal(output, expected_output)
    print("MaxPool2D SAME padding test passed!")

def test_maxpool2d_projection_valid():
    key = jax.random.PRNGKey(0)
    input_data = jax.random.normal(key, (1, 5, 5, 1))
    z = ops.maxpool_op(input_data, pool_size=(2, 2), strides=(2, 2), padding="VALID")
    (a_proj,) = ops.maxpool_proj(input_data, z, pool_size=(2, 2), strides=(2, 2), padding="VALID")
    z_proj = ops.maxpool_op(a_proj, pool_size=(2, 2), strides=(2, 2), padding="VALID")
    np.testing.assert_allclose(z_proj, z, atol=1e-2)


def test_maxpool2d_projection_same():
    key = jax.random.PRNGKey(1)
    input_data = jax.random.normal(key, (1, 3, 3, 1))
    z = ops.maxpool_op(input_data, pool_size=(2, 2), strides=(2, 2), padding="SAME")
    (a_proj,) = ops.maxpool_proj(input_data, z, pool_size=(2, 2), strides=(2, 2), padding="SAME")
    z_proj = ops.maxpool_op(a_proj, pool_size=(2, 2), strides=(2, 2), padding="SAME")
    np.testing.assert_allclose(z_proj, z, atol=1e-2)


if __name__ == "__main__":
    test_maxpool2d_valid()
    test_maxpool2d_same()
    test_maxpool2d_projection_valid()
    test_maxpool2d_projection_same()
