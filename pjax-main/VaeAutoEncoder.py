import jax
import jax.numpy as jnp
import pjax
from pjax import nn, optim
from data import MNISTDataModule
import matplotlib.pyplot as plt

dataset = MNISTDataModule(batch_size=20)
train_data = dataset.train_dataloader()
randkey = jax.random.PRNGKey(0)

def get_key():
    global randkey
    randkey, subkey = jax.random.split(randkey)
    return subkey

# 1. Define the autoencoder model
class VaeAutoencoder(nn.Module):
    def __init__(self, in_features, hidden_features, latent_features):
        super().__init__()
        self.latent_features = latent_features
        # Encoder
        self.encoder_dense1 = nn.Linear(in_features, hidden_features)
        self.encoder_relu = nn.ReLU(hidden_features)
        self.normalOut = nn.NormalLayer(hidden_features, latent_features)
        
        # Decoder
        self.decoder_dense1 = nn.Linear(latent_features, hidden_features)
        self.decoder_relu = nn.ReLU(hidden_features)
        self.decoder_dense2 = nn.Linear(hidden_features, in_features)

    def encode(self, x):
        x = self.encoder_dense1(x)
        x = self.encoder_relu(x)
        mean, logvar = self.normalOut(x)
        return mean, logvar

    def decode(self, z):
        x = self.decoder_dense1(z)
        x = self.decoder_relu(x)
        x = self.decoder_dense2(x)
        return x
    
    def __call__(self, x, encode_only=False, decode_only=False):
        # Encode
        if not decode_only:
            mean, sigma = self.encode(x)
            x = pjax.reparameterize(mean, sigma)
            x = pjax.identity(x)
            
        if encode_only:
            return mean, sigma
        
        if decode_only:
            return self.decode(x)
        
        return self.decode(x), mean, sigma
    
    
# 2. Initialize models and optimizers
key = jax.random.key(0)
model = VaeAutoencoder(in_features=784, hidden_features=256, latent_features=64)

params = model.init(key)

optimizer = optim.AlternatingProjections(steps_per_update=5)

# 3. Define training steps
@jax.jit
def train_step(params, x):
    x = jnp.array(x)
    def apply_fn(params):
        reconstruction = model.apply(params, x)[0]
        #jax.debug.print("reconstruction shape: {}", reconstruction.shape)
        return pjax.means_squared_error(reconstruction, x)    
    
    updated_params, loss = optimizer.update(apply_fn, params)
    return updated_params, loss

# 4. Training loop
losses = []

for step in range(150):
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
reconstructed = model.apply(params, test_images_flat)[0]

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

# 8. VAE-specific visualizations
from sklearn.decomposition import PCA
import numpy as np

# Collect all test encodings and labels for latent space visualization
all_encodings = []
all_labels = []

for batch_x, batch_y in train_data:
    batch_x_flat = jnp.array(batch_x.reshape(batch_x.shape[0], -1))
    mean = model.apply(params, batch_x_flat)[1]  
    all_encodings.append(np.array(mean))
    all_labels.append(np.array(batch_y))
    if len(all_encodings) >= 5:  # Use enough batches for visualization
        break

encodings = np.concatenate(all_encodings, axis=0)
labels = np.concatenate(all_labels, axis=0)

# Visualize latent space (2D projection via PCA)
pca = PCA(n_components=2)
encodings_2d = pca.fit_transform(encodings)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(encodings_2d[:, 0], encodings_2d[:, 1], c=labels, cmap='tab10', s=30, alpha=0.6)
plt.colorbar(scatter, label='Digit Class')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.title('VAE Latent Space (PCA projection)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 9. Generate samples from the prior (standard normal distribution)
n_samples = 16
eps = jax.random.normal(get_key(), shape=(n_samples, 64))

generated_samples = model.apply(params, eps, decode_only=True)

fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i in range(n_samples):
    row = i // 8
    col = i % 8
    axes[row, col].imshow(jnp.clip(generated_samples[i].reshape(28, 28), 0, 1), cmap='gray')
    axes[row, col].set_title(f'Generated {i+1}')
    axes[row, col].axis('off')

plt.suptitle('VAE: Samples Generated from Prior N(0, I)')
plt.tight_layout()
plt.show()

# 10. Interpolation in latent space
z1 = jax.random.normal(get_key(), shape=(1, model.latent_features))
z2 = jax.random.normal(get_key(), shape=(1, model.latent_features))

alpha_steps = jnp.linspace(0, 1, 10)
interpolations = jnp.array([model.apply(params, z1 * (1 - a) + z2 * a, decode_only=True) for a in alpha_steps])

fig, axes = plt.subplots(1, 10, figsize=(16, 2))
for i, interp in enumerate(interpolations):
    axes[i].imshow(jnp.clip(interp[0].reshape(28, 28), 0, 1), cmap='gray')
    axes[i].set_title(f'α={alpha_steps[i]:.1f}')
    axes[i].axis('off')

plt.suptitle('VAE: Latent Space Interpolation')
plt.tight_layout()
plt.show()

print("VAE training and visualization complete!")