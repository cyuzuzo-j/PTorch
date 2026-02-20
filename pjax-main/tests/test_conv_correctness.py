
import jax
import jax.numpy as jnp
import pytest
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from pjax.core.no_ops import conv_patch_transform, conv_patch_inverse

def test_conv_patch_identity_same_padding():
    """Test that conv_patch_inverse is the inverse of conv_patch_transform for SAME padding."""
    N = 2
    H, W = 16, 16
    C = 4
    kernel_shape = (3, 3)
    strides = (1, 1)
    # Using explicit padding tuple as recognized by current implementation for now
    # The implementation plan mentioned handling "SAME" or ensuring callers pass tuples.
    # pjax seems to rely on the caller for padding tuples in conv_patch_inverse currently.
    # Let's use the tuple that corresponds to SAME for 3x3 kernel: ((1,1), (1,1))
    padding = ((1, 1), (1, 1)) 
    
    key = jax.random.PRNGKey(42)
    min_val, max_val = -1.0, 1.0
    a = jax.random.uniform(key, (N, H, W, C), minval=min_val, maxval=max_val)
    
    # Transform
    z = conv_patch_transform(a, kernel_shape=kernel_shape, strides=strides, padding=padding)
    
    # Inverse
    a_rec = conv_patch_inverse(a, z, kernel_shape=kernel_shape, strides=strides, padding=padding)
    
    # Check shape
    assert a_rec.shape == a.shape
    
    # Check values
    # Note: reconstruction might not be exact identity if 'a' is high frequency and we are averaging?
    # Actually, conv_patch_transform just extracts patches. 
    # If we modify nothing and put them back, we are doing a "fold" operation.
    # If patches overlap, `conv_patch_inverse` averages them.
    # If the original image 'a' is consistent with the patches (which it is, because we extracted them from it),
    # then averaging the overlaps should return the exact same value.
    # Example: pixel (i,j) contributes to K patches. We sum K copies of pixel (i,j) and divide by K.
    # So it should be identity.
    
    diff = jnp.mean(jnp.abs(a_rec - a))
    max_diff = jnp.max(jnp.abs(a_rec - a))
    print(f"Mean diff: {diff}, Max diff: {max_diff}")
    
    assert jnp.allclose(a_rec, a, atol=1e-5)

def test_conv_patch_identity_valid_padding():
    """Test that conv_patch_inverse is the inverse of conv_patch_transform for VALID padding."""
    N = 2
    H, W = 16, 16
    C = 4
    kernel_shape = (3, 3)
    strides = (1, 1)
    padding = ((0, 0), (0, 0)) # VALID
    
    # For VALID padding, the inverse cannot reconstruct the edges if they were cut off.
    # conv_patch_transform with VALID padding produces smaller output.
    # conv_patch_inverse must reconstruct the *original* shape?
    # The current implementation of conv_patch_inverse takes 'a' as input (probably for shape reference).
    # If we use VALID padding, the border pixels of 'a' are not in 'z'.
    # So conv_patch_inverse will likely return 0s at the borders or NaNs if it tries to normalize by 0 count.
    
    key = jax.random.PRNGKey(123)
    a = jax.random.normal(key, (N, H, W, C))
    
    z = conv_patch_transform(a, kernel_shape=kernel_shape, strides=strides, padding=padding)
    
    # We expect reconstruction to be valid only in the center
    a_rec = conv_patch_inverse(a, z, kernel_shape=kernel_shape, strides=strides, padding=padding)
    
    # 3x3 VALID means we lose 1 pixel on each border
    # a_rec should match a in a[1:-1, 1:-1, :]
    
    a_center = a[:, 1:-1, 1:-1, :]
    a_rec_center = a_rec[:, 1:-1, 1:-1, :]
    
    assert jnp.allclose(a_rec_center, a_center, atol=1e-5)

if __name__ == "__main__":
    test_conv_patch_identity_same_padding()
    test_conv_patch_identity_valid_padding()
    print("All tests passed!")
