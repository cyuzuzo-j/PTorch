import jax
import jax.numpy as jnp
import pjax
from pjax import nn, optim
import numpy as np

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # in_features_x=28, in_features_y=28, in_channels=1, out_channels=1, kernel_shape=3
        self.conv = nn.FftConv2D(28, 28, 1, 1, 2)

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
        x = jax.random.normal(key, (1, 28, 28, 1))
        # Target: (batch=1, H=28, W=28, C=1)
        # We want the output to match this target
        y = jax.random.normal(key, (1, 28, 28, 1))

        # Define optimizer
        optimizer = optim.AlternatingProjections(steps_per_update=1)

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
        print(f"Initial Loss: {initial_loss}")

        # Initial kernel
        initial_kernel = params['conv.kernel']
        print(f"Initial Kernel (first 5 values): {initial_kernel.flatten()[:5]}")

        # Run one update step
        print("Running optimization step...")
        updated_params, loss_metric = train_step(params, x, y)
        
        # Check if params changed
        updated_kernel = updated_params['conv.kernel']
        print(f"Updated Kernel (first 5 values): {updated_kernel.flatten()[:5]}")
        
        kernel_diff = jnp.linalg.norm(updated_kernel - initial_kernel)
        print(f"Kernel difference norm: {kernel_diff}")

        # Check if loss decreased
        final_output = model.apply(updated_params, x)
        final_loss = jnp.mean((final_output - y) ** 2)
        print(f"Final Loss: {final_loss}")

        if final_loss >= initial_loss:
            print("Test Failed: Loss did not decrease.")
            break

if __name__ == "__main__":
    test_cnn_update()
