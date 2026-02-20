import sys
import os
sys.path.append(os.getcwd())

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
import pjax
from pjax.nn.modules import FftConv2D
from pjax import nn

def test_fft_multichannel():
    print("Testing FftConv2D with multiple channels...")
    key = random.PRNGKey(0)
    key, k1, k2 = random.split(key, 3)

    H, W = 8, 8
    kernel_size = 3
    in_channels = 2
    out_channels = 3
    batch_size = 2

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
    input_shape = (batch_size, H, W, in_channels) 
    input_data = random.normal(k2, input_shape)
    
    print(f"Input shape: {input_data.shape}")
    
    output_fft_conv = model.apply(params, input_data)
    print("FftConv2D execution successful.")
    print(f"Output shape: {output_fft_conv.shape}")
    
    expected_shape = (batch_size, H, W, out_channels)
    if output_fft_conv.shape == expected_shape:
        print("Output shape matches expected shape.")
    else:
        print(f"Output shape mismatch! Expected {expected_shape}, got {output_fft_conv.shape}")
        sys.exit(1)

    # Verify kernel shape
    kernel_val = params['cnn.kernel']
    print(f"Kernel shape: {kernel_val.shape}")
    expected_kernel_shape = (kernel_size, kernel_size, in_channels, out_channels)
    if kernel_val.shape == expected_kernel_shape:
        print("Kernel shape matches expected shape.")
    else:
        print(f"Kernel shape mismatch! Expected {expected_kernel_shape}, got {kernel_val.shape}")
        sys.exit(1)

if __name__ == "__main__":
    test_fft_multichannel()
