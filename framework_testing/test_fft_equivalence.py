import sys
import os
sys.path.append(os.getcwd())

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
import pjax
from pjax.nn.modules import FftConv2D
from pjax import nn, optim
def test_fft_conv_equivalence():
    print("Testing FftConv2D equivalence...")
    key = random.PRNGKey(0)
    key, k1, k2 = random.split(key, 3)

    H, W = 10, 10
    kernel_size = 3
    in_channels = 1
    out_channels = 1

    # Instantiate FftConv2D
    class CNN_pjax(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = FftConv2D(
                in_features_x=H,
                in_features_y=W,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_shape=kernel_size
)
        def __call__(self, x):
            return self.cnn(x)
    model = CNN_pjax()
    # Initialize parameters
    params = model.init(k1)
    
    # Create input
    # Docstring says (..., H, W, C)
    input_shape = (1, H, W, 1) 
    input_data = random.normal(k2, input_shape)
    
    print(f"Input shape: {input_data.shape}")
    
    output_fft_conv = model.apply(params, input_data)
    print("FftConv2D execution successful.")
    print(f"Output shape: {output_fft_conv.shape}")

    # Extract kernel
    kernel_val = params['cnn.kernel']
    print(f"Kernel shape: {kernel_val.shape}")

    # Prepare data for comparison
    # We need 2D arrays for convolve2d
    if input_data.ndim == 4:
        input_2d = input_data[0, :, :, 0]
        out_fft_val = output_fft_conv[0, :, :, 0]
    elif input_data.ndim == 2:
        input_2d = input_data
        out_fft_val = output_fft_conv
    else:
        print("Unexpected input/output dimensions.")
        return
        
    
    # Reference 2: Standard convolve2d 'same', 'wrap'
    # This centers the kernel.
    # jax.scipy.signal.convolve2d only supports boundary='fill' (zero padding).
    # To simulate 'wrap' (circular convolution), we can pad the input manually with wrap mode.
    
    # Pad input to handle wrapping
    pad_h = kernel_size // 2
    pad_w = kernel_size // 2
    input_padded = jnp.pad(input_2d, ((pad_h, pad_h), (pad_w, pad_w)), mode='wrap')
    
    # Use 'valid' mode on padded input to get the same size as original input
    out_ref_wrap = jax.scipy.signal.convolve2d(input_padded, kernel_val, mode='valid')
    
    diff_ref_wrap = jnp.mean(jnp.abs(out_fft_val - out_ref_wrap))
    print(f"Diff with convolve2d(mode='same', boundary='wrap'): {diff_ref_wrap}")

    
    
if __name__ == "__main__":
    test_fft_conv_equivalence()
