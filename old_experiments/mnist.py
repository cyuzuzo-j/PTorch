import jax.numpy as np
import jax.random as random
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import tools.optimize as optimize
import tools.projections as projections
from data import MNISTDataModule
import matplotlib.pyplot as plt

rand_key = random.key(123)

# Setup problem
inputDim = 784  # 28x28 MNIST images
hiddenDim = 256
latentDim = 64
samples = 128

dataset = MNISTDataModule(batch_size=samples)
train_data = dataset.train_dataloader()

optimizer = optimize.DouglassRachford

# Initialize weight matrices
# Encoder: input -> hidden -> latent
W_enc1 = random.normal(rand_key, (hiddenDim, inputDim))
W_enc1 = np.concatenate((W_enc1, np.eye(hiddenDim)), axis=1)  # add bias row
W_enc2 = random.normal(rand_key, (latentDim, hiddenDim + 1))

# Decoder: latent -> hidden -> output
W_dec1 = random.normal(rand_key, (hiddenDim, latentDim))
W_dec1 = np.concatenate((W_dec1, np.eye(hiddenDim)), axis=1)  # add bias row
W_dec2 = random.normal(rand_key, (inputDim, hiddenDim + 1))

b_enc1 = random.normal(rand_key, (hiddenDim,))
b_enc2_val = 1
b_dec1 = random.normal(rand_key, (hiddenDim,))
b_dec2_val = 1

# Setup optimizers for each layer
optEnc1 = optimizer([projections.stepActivation], [projections.bilinearMatrix])
optEnc2 = optimizer([], [projections.bilinearMatrix])
optDec1 = optimizer([projections.stepActivation], [projections.bilinearMatrix])
optDec2 = optimizer([], [projections.bilinearMatrix])

error = []

# Training loop
for epoch in range(2):
    for batch_idx, (x, y) in enumerate(train_data):
        x_batch = x.reshape(x.shape[0], -1)  # Flatten images
        batchError = np.zeros(x_batch.shape[0])
        
        for k, x_sample in enumerate(x_batch):
            # Forward pass - Encoder
            x_in = x_sample
            x_in_aug = np.append(x_in, b_enc1)
            x_hidden_enc = W_enc1 @ x_in_aug
            h_hidden_enc = np.maximum(x_hidden_enc, 0)  # ReLU
            h_hidden_enc_aug = np.append(h_hidden_enc, b_enc2_val)
            x_latent = W_enc2 @ h_hidden_enc_aug
            
            # Forward pass - Decoder
            x_latent_aug = np.append(x_latent, b_dec1)
            x_hidden_dec = W_dec1 @ x_latent_aug
            h_hidden_dec = np.maximum(x_hidden_dec, 0)  # ReLU
            h_hidden_dec_aug = np.append(h_hidden_dec, b_dec2_val)
            x_recon = W_dec2 @ h_hidden_dec_aug
            
            # Calculate reconstruction error
            recon_error = np.sum((x_recon - x_in) ** 2)
            batchError = batchError.at[k].set(recon_error)
            
            # Backward pass with projections
            target = x_in
            x_recon, w_dec2_proj, _ = optDec2.step_layer(x_recon, W_dec2, target)
            h_hidden_dec_aug, w_dec1_proj, x_hidden_dec = optDec1.step_layer(h_hidden_dec_aug, W_dec1, x_hidden_dec)
            h_hidden_dec = h_hidden_dec_aug[:][:-1]
            
            x_latent_aug, w_enc2_proj, x_latent = optEnc2.step_layer(x_latent_aug, W_enc2, x_latent)
            h_hidden_enc_aug, w_enc1_proj, x_hidden_enc = optEnc1.step_layer(h_hidden_enc_aug, W_enc1, x_hidden_enc)
            h_hidden_enc = h_hidden_enc_aug[:][:-1]
            
            # Update weights and biases
            b_dec2_val = b_dec2_val + (1/(k+1)) * (h_hidden_dec_aug[:][-1:] - b_dec2_val)
            b_dec1 = b_dec1 + (1/(k+1)) * (x_latent_aug[:][-len(b_dec1):] - b_dec1)
            b_enc2_val = b_enc2_val + (1/(k+1)) * (h_hidden_enc_aug[:][-1:] - b_enc2_val)
            b_enc1 = b_enc1 + (1/(k+1)) * (x_in_aug[:][-len(b_enc1):] - b_enc1)
            
            W_dec2 = W_dec2 + (1/(k+1)) * (w_dec2_proj - W_dec2)
            W_dec1 = W_dec1 + (1/(k+1)) * (w_dec1_proj - W_dec1)
            W_enc2 = W_enc2 + (1/(k+1)) * (w_enc2_proj - W_enc2)
            W_enc1 = W_enc1 + (1/(k+1)) * (w_enc1_proj - W_enc1)
        
        error.append(np.mean(batchError))
        print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {error[-1]:.6f}")
        
        if batch_idx == 4:
            break

# Visualize convergence
plt.figure(figsize=(12, 6))
plt.plot(error, linewidth=2)
plt.xlabel('Batch Step')
plt.ylabel('Reconstruction Loss (MSE)')
plt.title('Autoencoder Training Convergence (Projection-based)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Final Loss: {error[-1]:.6f}")
