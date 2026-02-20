import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import jax
import jax.numpy as jnp
import pjax
from pjax import nn, optim
import numpy as np
import matplotlib.pyplot as plt

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # in_features_x=28, in_features_y=28, in_channels=1, out_channels=1, kernel_shape=3
        self.conv = nn.FftConv2D(4, 4, 1, 1, 4)

    def __call__(self, x):
        return self.conv(x)

def test_cnn_update():
    key = jax.random.key(0)
    print("Starting CNN update test...")
    while True:
        # Initialize model
        model = SimpleCNN()
        key, subkey = jax.random.split(key)
        params = model.init(key)

        # Create dummy data
        # Input: (batch=1, H=28, W=28, C=1)
        x = jax.random.normal(key, (1, 4, 4, 1))
        # Target: (batch=1, H=28, W=28, C=1)
        # We want the output to match this target
        y = jax.random.normal(key, (1, 4, 4, 1))*100

        # Define optimizer
        optimizer = optim.DouglasRachford(steps_per_update=1)

        # Define training step
        @jax.jit
        def train_step(params, x, y):
            def apply_fn(params):
                x_complex = x.astype(jax.numpy.complex64)
                output = model.apply(params, x_complex)
                y_complex = y.astype(output.dtype)
                return pjax.means_squared_error(output, y_complex)

            updated_params, loss = optimizer.update(apply_fn, params)
            return updated_params, loss

        # Initial loss
        initial_output = model.apply(params, x)
        initial_loss = jnp.mean((initial_output - y) ** 2)

        # Initial kernel
        initial_kernel = params['conv.kernel']

        # Run one update step
        
        params, loss_metric = train_step(params, x, y)
        
        # Check if params changed
        updated_kernel = params['conv.kernel']
        
        kernel_diff = jnp.linalg.norm(updated_kernel - initial_kernel)

        # Check if loss decreased
        final_output = model.apply(params, x)
        final_loss = jnp.mean((final_output - y) ** 2)

        if  jnp.abs(final_loss) > jnp.abs(initial_loss):
            print("Test Failed: Loss Increased from", jnp.abs(initial_loss), "to", jnp.abs(final_loss))
            break

if __name__ == "__main__":
    test_cnn_update()
