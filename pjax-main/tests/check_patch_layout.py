import jax
import jax.numpy as jnp

N = 1
H, W = 4, 4
C = 1
a = jnp.arange(16, dtype=jnp.float32).reshape(N, H, W, C)

patch = jax.lax.conv_general_dilated_patches(
    a,
    filter_shape=(2, 2),
    window_strides=(1, 1),
    padding="VALID",
    dimension_numbers=("NHWC", "HWIO", "NHWC"),
)

print("Original array:")
print(a[0, :, :, 0])
print("\nPatch output shape:", patch.shape)
print("First patch at (0, 0):")
print(patch[0, 0, 0, :])

# Now with C = 2
C2 = 2
a2 = jnp.arange(32, dtype=jnp.float32).reshape(N, H, W, C2)
patch2 = jax.lax.conv_general_dilated_patches(
    a2,
    filter_shape=(2, 2),
    window_strides=(1, 1),
    padding="VALID",
    dimension_numbers=("NHWC", "HWIO", "NHWC"),
)

print("\nFirst patch with C=2:")
print(patch2[0, 0, 0, :])

