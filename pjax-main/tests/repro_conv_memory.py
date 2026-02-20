
import jax
import jax.numpy as jnp
import time
import functools
from pjax.core.no_ops import conv_patch_transform, conv_patch_inverse

def benchmark_conv_inverse():
    # Setup parameters
    N = 8
    H, W = 64, 64
    C = 64
    kernel_shape = (3, 3)
    strides = (1, 1)
    padding = ((1, 1), (1, 1))
    
    # Create input
    key = jax.random.PRNGKey(0)
    input_shape = (N, H, W, C)
    a = jax.random.normal(key, input_shape)
    
    # Do transform to get z
    print("Running conv_patch_transform...")
    z = conv_patch_transform(a, kernel_shape=kernel_shape, strides=strides, padding=padding)
    print(f"z shape: {z.shape}")
    
    # Benchmark inverse
    print("Benchmarking conv_patch_inverse...")
    
    # Force compilation
    start = time.time()
    out = conv_patch_inverse(a, z, kernel_shape=kernel_shape, strides=strides, padding=padding)
    out.block_until_ready()
    print(f"Compilation + First run time: {time.time() - start:.4f}s")
    
    # Run multiple times
    times = []
    for _ in range(10):
        start = time.time()
        out = conv_patch_inverse(a, z, kernel_shape=kernel_shape, strides=strides, padding=padding)
        out.block_until_ready()
        times.append(time.time() - start)
        
    print(f"Average execution time: {sum(times)/len(times):.4f}s")
    
    # Check correctness (identity check)
    # Note: conv_patch_inverse(a, z) actually assumes z is modified, but if we pass the extracted patches back, 
    # it should approximately reconstruct 'a' IF 'a' is smooth? 
    # Wait, conv_patch_inverse(a, z) returns an approximation or exact?
    # It seems to be averaging overlapping patches. If patches are unmodified from 'a', 
    # and we average them back, we should get 'a' exactly (modulo floating point).
    
    diff = jnp.mean(jnp.abs(out - a))
    print(f"Reconstruction difference (L1 mean): {diff:.6f}")
    
    # Memory profiling is hard from within script without proper tools, 
    # but we can look at HLO or just rely on the user's report and execution time.

if __name__ == "__main__":
    benchmark_conv_inverse()
