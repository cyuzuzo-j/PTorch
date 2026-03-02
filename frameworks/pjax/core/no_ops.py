"""Shape transformation operations that don't require projection operators.

Each transformation implements forward and inverse operations:
    ``transform(input, /, *, **kwargs) -> transformed_output``
    ``inverse(input, transformed_output, /, *, **kwargs) -> original_input``

These are invertible shape manipulations that preserve data while changing
tensor layout or structure. Used internally by the computation graph.
"""

import jax
import numpy as np
from jax import numpy as jnp

from .computation import make_shape_transform


def batchnorm_transform(a, /, *, eps=1e-5):
    """Batch normalization as a parameter-free shape transform.

    Normalizes ``a`` across all dimensions except the last (features)::

        output = (a - mean) / sqrt(var + eps)

    Args:
        a: input array of shape ``(N, ..., features)``.
        eps: small constant for numerical stability.

    Returns:
        Normalized array with the same shape as ``a``.
    """
    reduce_axes = tuple(range(a.ndim - 1))
    mean = jnp.mean(a, axis=reduce_axes, keepdims=True)
    var = jnp.var(a, axis=reduce_axes, keepdims=True)
    return (a - mean) / jnp.sqrt(var + eps)


def batchnorm_inverse(a, z, /, *, eps=1e-5):
    """Inverse of batch normalization: denormalize using original statistics.

    Restores from normalized space back to input space using the mean and
    variance computed from the original input ``a``.
    """
    reduce_axes = tuple(range(a.ndim - 1))
    mean = jnp.mean(a, axis=reduce_axes, keepdims=True)
    var = jnp.var(a, axis=reduce_axes, keepdims=True)
    return z * jnp.sqrt(var + eps) + mean


batchnorm = make_shape_transform(
    "batchnorm", transform=batchnorm_transform, inverse=batchnorm_inverse
)


def index_transform(a, idx, /):
    """Index a tensor."""
    assert isinstance(a, jnp.ndarray)
    return a[idx]


def index_inverse(a, idx, z, /):
    return a.at[idx].set(z)


index = make_shape_transform(
    "index",
    transform=index_transform,
    inverse=index_inverse,
)


def advanced_index_transform(a, /, *, idx):
    """Advanced indexing of a tensor.

    Here, ``idx`` can be any valid JAX indexing expression (e.g., a slice).
    Note it is a keyword argument, so for different indexing expressions, recompilation is needed.
    """
    assert isinstance(a, jnp.ndarray)
    return a[idx]


def advanced_index_inverse(a, z, /, *, idx):
    """Inverse of advanced indexing: restores the original tensor."""
    return a.at[idx].set(z)


advanced_index = make_shape_transform(
    "advanced_index", transform=advanced_index_transform, inverse=advanced_index_inverse
)


def reshape_transform(a, /, *, shape):
    """Reshape a tensor to a given shape."""
    return jnp.reshape(a, shape)


def reshape_inverse(a, z, /, *, shape):
    """Inverse of reshape transform: restores the original shape."""
    return jnp.reshape(z, a.shape)


reshape = make_shape_transform("reshape", transform=reshape_transform, inverse=reshape_inverse)


def transpose_transform(a, /, *, axes):
    """Transpose a tensor along specified axes."""
    return jnp.transpose(a, axes)


def transpose_inverse(a, z, /, *, axes):
    """Inverse of transpose transform: reverses the transposition."""
    return jnp.transpose(z, np.argsort(axes))


transpose = make_shape_transform("transpose", transform=transpose_transform, inverse=transpose_inverse)


def repeat_transform(a, /, *, repeats, axis):
    """Repeat elements of a tensor along a specified axis."""
    return jnp.repeat(a, repeats, axis=axis)


def repeat_inverse(a, z, /, *, repeats, axis):
    """Inverse of repeat transform: averages the repeated values."""
    print(a.shape,z.shape)
    axis = axis if axis >= 0 else a.ndim + axis
    assert a.shape[axis] * repeats == z.shape[axis]
    reshaped = z.reshape(z.shape[:axis] + (repeats, -1) + z.shape[axis + 1 :])
    return reshaped.mean(axis=axis)


repeat = make_shape_transform("repeat", transform=repeat_transform, inverse=repeat_inverse)


def concatenate_transform(*args, axis=0):
    """Concatenate multiple tensors along a specified axis."""
    return jnp.concatenate(args, axis=axis)


def concatenate_inverse(*args, axis=0):
    """Inverse of concatenate transform: splits the tensor back into original parts."""
    args, z = args[:-1], args[-1]
    split_indices = np.cumsum([x.shape[axis] for x in args[:-1]])
    return jnp.split(z, split_indices, axis=axis)

concatenate = make_shape_transform("concatenate", transform=concatenate_transform, inverse=concatenate_inverse)


def zero_pad_transform(a, /, *, pad_width):
    """Apply zero-padding to a tensor."""
    return jnp.pad(a, pad_width, mode="constant")


def zero_pad_inverse(a, z, /, *, pad_width):
    """Inverse of zero-pad transform: removes padding."""
    return z[tuple(slice(i, -j if j else None) for i, j in pad_width)]


zero_pad = make_shape_transform("zero_pad", transform=zero_pad_transform, inverse=zero_pad_inverse)


def conv_patch_transform(a, /, *, kernel_shape, strides, padding):
    """Extract patches from an array for convolution-like operations.

    Returns a tensor of shape ``(N, H_out, W_out, C_in * H_k * W_k)``,
    where ``H_out`` and ``W_out`` are the output height and width after convolution,
    and ``H_k`` and ``W_k`` are the kernel height and width.
    """
    return jax.lax.conv_general_dilated_patches(
        a,
        filter_shape=kernel_shape,
        window_strides=strides,
        padding=padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )


def conv_patch_inverse(a, z, /, *, kernel_shape, strides, padding):
    """Inverse of convolution patch extraction: reconstructs the original array.
    
    Optimized implementation using scatter_add instead of conv_transpose.
    """
    N, H_in, W_in, C_in = a.shape
    _, H_out, W_out, _ = z.shape
    H_k, W_k = kernel_shape
    
    # Parse padding
    # We assume 'padding' is ((top, bottom), (left, right)).
    if isinstance(padding, str):
         # If "SAME" or "VALID", we should compute explicit padding.
         # For now, raise error or rely on user passing tuple as per plan.
         # But wait, existing code might pass "SAME"? 
         # The benchmark passed "SAME" before I fixed it.
         # Let's add basic support for SAME/VALID strings if possible, 
         # but actually jax.lax.conv_general_dilated_patches handles strings.
         # For inverse, we need explicit padding to map indices correctly.
         # We can't easily reverse "SAME" without knowing input shape (which we have: a.shape).
         # So we could compute it. But for safety, let's enforce tuple or try to compute.
         pass
         
    if isinstance(padding, str):
        # Very rough fallback or error
        # Assuming H_out was computed correctly by patches:
        # H_out = (H_in + pad_total - H_k) // stride + 1
        # It's hard to distinguish left/right padding from just "SAME".
        # But usually "SAME" means total padding s.t. output size is ceil(H_in/stride).
        # Let's error out for now as agreed in plan constraints.
        raise ValueError("conv_patch_inverse optimized requires explicit padding tuples.")
        
    (pad_top, _), (pad_left, _) = padding

    # 1. Coordinate grids
    h_out_idx = jnp.arange(H_out)
    w_out_idx = jnp.arange(W_out)
    h_k_idx = jnp.arange(H_k)
    w_k_idx = jnp.arange(W_k)
    
    # Meshgrid including kernel dimensions
    # Shape: (H_out, W_out, H_k, W_k)
    hh, ww, hk, wk = jnp.meshgrid(h_out_idx, w_out_idx, h_k_idx, w_k_idx, indexing='ij')
    
    # Compute input coordinates
    stride_h, stride_w = strides
    h_in_idx = hh * stride_h + hk - pad_top
    w_in_idx = ww * stride_w + wk - pad_left
    
    # 2. Reshape z and transpose for scattering
    # Input z: (N, H_out, W_out, C_in * H_k * W_k) -> reshape to separate dims
    
    z_reshaped = z.reshape(N, H_out, W_out, C_in, H_k, W_k)
    
    # We want to add z values to target at (n, c, h_in, w_in).
    
    # Transpose z to (N, C_in, H_out, W_out, H_k, W_k) to align with target (N, C_in, H, W)
    z_tr = z_reshaped.transpose(0, 3, 1, 2, 4, 5)
    
    # Initialize target and count buffers
    # (N, C_in, H_in, W_in)
    target = jnp.zeros((N, C_in, H_in, W_in), dtype=z.dtype)
    counts = jnp.zeros((N, C_in, H_in, W_in), dtype=z.dtype)
    
    # Handle negative indices (padding) wrapping around in JAX .at[] indexing
    valid_h = (h_in_idx >= 0) & (h_in_idx < H_in)
    valid_w = (w_in_idx >= 0) & (w_in_idx < W_in)
    valid_mask = valid_h & valid_w
    
    # Expand valid_mask to broadcast with z_tr: (1, 1, H_out, W_out, H_k, W_k)
    valid_mask_exp = jnp.expand_dims(valid_mask, axis=(0, 1))
    
    z_tr = jnp.where(valid_mask_exp, z_tr, jnp.array(0.0, dtype=z.dtype))
    count_inc = jnp.where(valid_mask_exp, jnp.array(1.0, dtype=z.dtype), jnp.array(0.0, dtype=z.dtype))
    
    # Send invalid indices to 0 safely (they will add 0 due to the mask above)
    h_in_idx_safe = jnp.where(valid_mask, h_in_idx, 0)
    w_in_idx_safe = jnp.where(valid_mask, w_in_idx, 0)
    
    # Use index_add (scatter_add) with broadcasting.
    target = target.at[:, :, h_in_idx_safe, w_in_idx_safe].add(z_tr)
    counts = counts.at[:, :, h_in_idx_safe, w_in_idx_safe].add(count_inc)
    
    # Safe division
    counts = jnp.maximum(counts, 1.0)
    out = target / counts
    
    # Return to NHWC
    return out.transpose(0, 2, 3, 1)

def detach_complex_transform(a, /):
    """Detach complex tensor by separating real and imaginary parts.
       Add the imaginary part as an additional channel.
    """
    real_part = jnp.real(a)
    imag_part = jnp.imag(a)
    
    return jnp.concatenate([real_part, imag_part], axis=1)


def detach_complex_inverse(a, z,/):
    """Inverse of detach complex: recombines real and imaginary parts."""
    half = z.shape[1] //2
    real_part = z[:, :half, ...]
    imag_part = z[:, half:, ...]
    return real_part + 1j * imag_part

detach_complex = make_shape_transform(
    "detach_complex", transform=detach_complex_transform, inverse=detach_complex_inverse
)
attach_complex = make_shape_transform(
    "attach_complex", transform=lambda a: detach_complex_inverse(a,a), inverse= lambda a,z: detach_complex_transform(z)
)

def fft2d(*args):
    """Compute the 2D FFT of the last two dimensions of the input array."""
    a = args[0] if len(args) == 1 else args[1]
    return jnp.fft.fft2(a, axes=(-2, -1))

def ifft2d(*args):
    """Compute the 2D inverse FFT of the last two dimensions of the input array."""
    a = args[0] if len(args) == 1 else args[1]
    return jnp.fft.ifft2(a, axes=(-2, -1))

def inverse_ifft2d(*args):
    """Inverse of 2D FFT: computes the 2D inverse FFT."""
    return fft2d(*args)

def inverse_fft2d(*args):
    """Inverse of 2D FFT: computes the 2D inverse FFT."""
    return ifft2d(*args)

fourier = make_shape_transform("fft", transform=fft2d, inverse=inverse_fft2d)
inv_fourier = make_shape_transform("ifft", transform=ifft2d, inverse=inverse_ifft2d)
conv_patch = make_shape_transform("conv_patch", transform=conv_patch_transform, inverse=conv_patch_inverse)


def flip_transform(a, /, *, axis):
    """Flip array elements along specified axes."""
    if isinstance(axis, int):
        axis = (axis,)
    result = a
    for ax in axis:
        result = jnp.flip(result, axis=ax)
    return result


def flip_inverse(a, z, /, *, axis):
    """Inverse of flip transform: flip again (flip is self-inverse)."""
    return flip_transform(z, axis=axis)


flip = make_shape_transform("flip", transform=flip_transform, inverse=flip_inverse)


def roll_transform(a, /, *, shift, axis):
    """Roll array elements along a given axis."""
    return jnp.roll(a, shift, axis=axis)


def roll_inverse(a, z, /, *, shift, axis):
    """Inverse of roll transform: rolls back in the opposite direction."""
    # If shift is an int, -shift works.
    # If shift is a tuple, we need to negate each element.
    if isinstance(shift, tuple):
        neg_shift = tuple(-s for s in shift)
    else:
        neg_shift = -shift
    return jnp.roll(z, neg_shift, axis=axis)

roll = make_shape_transform("roll", transform=roll_transform, inverse=roll_inverse)
