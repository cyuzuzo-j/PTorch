
import jax
import jax.numpy as jnp
import time
from pjax.core.no_ops import conv_patch_inverse

def debug_conv():
    print("Starting debug_conv...")
    
    # Very small case
    N = 1
    H, W = 4, 4
    C = 1
    kernel_shape = (2, 2)
    strides = (1, 1)
    padding = ((0, 0), (0, 0)) # VALID
    
    H_out = 3
    W_out = 3
    
    a_shape = (N, H, W, C)
    z_shape = (N, H_out, W_out, C * kernel_shape[0] * kernel_shape[1])
    
    print(f"a_shape: {a_shape}")
    print(f"z_shape: {z_shape}")
    
    a = jnp.zeros(a_shape)
    z = jnp.ones(z_shape)
    
    print("Calling conv_patch_inverse (VALID)...")
    try:
        out = conv_patch_inverse(a, z, kernel_shape=kernel_shape, strides=strides, padding=padding)
        out.block_until_ready()
        print("Success VALID!")
        print(f"Output shape: {out.shape}")
    except Exception as e:
        print(f"Error VALID: {e}")

    # Case with SAME padding (potentially causing OOB)
    padding_same = ((1, 1), (1, 1))
    H_out_same = 5 # (4+2-2)//1 + 1 = 5? No.
    # Input 4. Pad 1+1=2. Total 6. Kernel 2. Stride 1. Out = (6-2)/1 + 1 = 5.
    
    z_same_shape = (N, 5, 5, C * 4)
    z_same = jnp.ones(z_same_shape)
    
    print("Calling conv_patch_inverse (SAME)...")
    try:
        out_same = conv_patch_inverse(a, z_same, kernel_shape=kernel_shape, strides=strides, padding=padding_same)
        out_same.block_until_ready()
        print("Success SAME!")
        print(f"Output shape: {out_same.shape}")
    except Exception as e:
        print(f"Error SAME: {e}")

if __name__ == "__main__":
    debug_conv()
