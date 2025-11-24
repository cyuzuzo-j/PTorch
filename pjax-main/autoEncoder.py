import jax
import jax.numpy as jnp
import pjax
from pjax import nn, optim
from data import MNISTDataModule
import matplotlib.pyplot as plt

dataset = MNISTDataModule(batch_size=16)
train_data = dataset.train_dataloader()
# 1. Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, in_features, hidden_features, latent_features):
        super().__init__()
        # Encoder
        self.encoder_dense1 = nn.Linear(in_features, hidden_features)
        self.encoder_relu = nn.ReLU(hidden_features)
        self.encoder_dense2 = nn.Linear(hidden_features, latent_features)
        
        # Decoder
        self.decoder_dense1 = nn.Linear(latent_features, hidden_features)
        self.decoder_relu = nn.ReLU(hidden_features)
        self.decoder_dense2 = nn.Linear(hidden_features, in_features)

    def __call__(self, x):
        # Encode
        x = self.encoder_dense1(x)
        x = self.encoder_relu(x)
        x = self.encoder_dense2(x)
        
        # Decode
        x = self.decoder_dense1(x)
        x = self.decoder_relu(x)
        x = self.decoder_dense2(x)
        return x

# 2. Initialize models and optimizers
key = jax.random.key(0)
model = Autoencoder(in_features=784, hidden_features=256, latent_features=16)

params = model.init(key)

optimizer = optim.DouglasRachfordMonumentum(steps_per_update=1)

# 3. Define training steps
@jax.jit
def train_step(params, x):
    def apply_fn(params):
        reconstruction = model.apply(params, x)
        return pjax.means_squared_error(reconstruction, x)    
    
    updated_params, loss = optimizer.update(apply_fn, params)
    return updated_params, loss

# 4. Training loop
losses = []

for step in range(5000):
    for i, (x, y) in enumerate(train_data):
        x = x.reshape(x.shape[0], -1)  # flatten the images
        print(f"Batch {i}")
        params, loss = train_step(params, x)
        losses.append(float(loss))
        if i == 0:
            break
    print(f"Step: {step}, Loss: {loss:.4f}")

# 5. Visualize convergence
plt.figure(figsize=(12, 6))
plt.plot(losses, label='AlternatingProjections', linewidth=2)
plt.xlabel('Batch Step')
plt.ylabel('Reconstruction Loss (MSE)')
plt.title('Autoencoder Training Convergence')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 6. Display reconstruction example
print(f"\nFinal Loss: {losses[-1]:.6f}")

# 7. Visualize reconstructions
# Get a batch of test images
test_batch = next(iter(train_data))
test_images = test_batch[0][:16]  # Get first 16 images
test_images_flat = test_images.reshape(test_images.shape[0], -1)

# Get reconstructions
reconstructed = model.apply(params, test_images_flat)

# Reshape back to image format
test_images_reshaped = test_images.reshape(-1, 28, 28)
reconstructed_reshaped = reconstructed.reshape(-1, 28, 28)

# Plot original vs reconstructed
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
for i in range(16):
    # Original images
    axes[i // 4, (i % 4) * 2].imshow(test_images_reshaped[i], cmap='gray')
    axes[i // 4, (i % 4) * 2].set_title('Original')
    axes[i // 4, (i % 4) * 2].axis('off')
    
    # Reconstructed images
    axes[i // 4, (i % 4) * 2 + 1].imshow(jnp.clip(reconstructed_reshaped[i], 0, 1), cmap='gray')
    axes[i // 4, (i % 4) * 2 + 1].set_title('Reconstructed')
    axes[i // 4, (i % 4) * 2 + 1].axis('off')

plt.suptitle('Autoencoder: Original vs Reconstructed Images')
plt.tight_layout()
plt.show()