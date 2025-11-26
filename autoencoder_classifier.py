import jax.numpy as np
import jax.random as random
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA

import projections
from optimize import DouglassRachford, AlternatingProjection
from data import MNISTDataModule


def step_activation(x):
    """Step activation function: 1 if x > 0, else 0."""
    return np.where(x > 0, 1.0, 0.0)


def he_initialization(key, shape, fan_in):
    """He initialization: samples from N(0, sqrt(2/fan_in))."""
    std = np.sqrt(2.0 / fan_in)
    return random.normal(key, shape) * std


def setup_autoencoder_problem(data_module, rand_key):
    """Initializes the autoencoder problem parameters using MNIST data."""
    keys = random.split(rand_key, 8)
    
    # --- Hyperparameters ---
    inputDim = 28 * 28  # MNIST flattened
    hiddenDim = 256
    latentDim = 64
    
    # --- Get first batch from MNIST training data ---
    train_loader = data_module.train_dataloader()
    
    # --- Weight Initialization (Encoder) with He Initialization ---
    W_enc_hidden = he_initialization(keys[0], (hiddenDim, inputDim), inputDim)
    W_enc_hidden = np.concatenate((W_enc_hidden, np.eye(hiddenDim)), axis=1)
    
    W_enc_latent = he_initialization(keys[1], (latentDim, hiddenDim), hiddenDim)
    W_enc_latent = np.concatenate((W_enc_latent, np.eye(latentDim)), axis=1)
    
    # --- Weight Initialization (Decoder) with He Initialization ---
    W_dec_hidden = he_initialization(keys[2], (hiddenDim, latentDim), latentDim)
    W_dec_hidden = np.concatenate((W_dec_hidden, np.eye(hiddenDim)), axis=1)
    
    W_dec_output = he_initialization(keys[3], (inputDim, hiddenDim), hiddenDim)
    W_dec_output = np.concatenate((W_dec_output, np.eye(inputDim)), axis=1)
    
    # --- Bias Initialization (zeros for He initialization) ---
    b_enc_hidden = np.zeros(hiddenDim)
    b_enc_latent = np.zeros(latentDim)
    b_dec_hidden = np.zeros(hiddenDim)
    b_dec_output = np.zeros(inputDim)
    
    problem_params = {
        "W_enc_hidden": W_enc_hidden, "b_enc_hidden": b_enc_hidden,
        "W_enc_latent": W_enc_latent, "b_enc_latent": b_enc_latent,
        "W_dec_hidden": W_dec_hidden, "b_dec_hidden": b_dec_hidden,
        "W_dec_output": W_dec_output, "b_dec_output": b_dec_output,
        "train_loader": train_loader,
    }
    return problem_params


def run_autoencoder_experiment(problem_params, num_iterations=500):
    """Runs autoencoder training using projection-based optimization with Step activation."""
    
    # --- Unpack parameters ---
    W_enc_hidden = problem_params["W_enc_hidden"]
    b_enc_hidden = problem_params["b_enc_hidden"]
    W_enc_latent = problem_params["W_enc_latent"]
    b_enc_latent = problem_params["b_enc_latent"]
    W_dec_hidden = problem_params["W_dec_hidden"]
    b_dec_hidden = problem_params["b_dec_hidden"]
    W_dec_output = problem_params["W_dec_output"]
    b_dec_output = problem_params["b_dec_output"]
    train_loader = problem_params["train_loader"]
    
    hiddenDim = W_enc_hidden.shape[0]
    latentDim = W_enc_latent.shape[0]
    
    # --- Initialize optimizers for each layer ---
    # Bilinear for weight layers (no activation)
    opt_enc_hidden = DouglassRachford([], [projections.bilinearMatrix])
    opt_enc_hidden_activation = DouglassRachford([projections.stepActivation], [])
    # Step activation for latent
    opt_enc_latent = DouglassRachford([projections.lowrankApproximation], [projections.bilinearMatrix])
    opt_enc_latent_activation = DouglassRachford([projections.stepActivation], [])
    # Bilinear for decoder hidden
    opt_dec_hidden = DouglassRachford([], [projections.bilinearMatrix])
    opt_dec_hidden_activation = DouglassRachford([projections.stepActivation], [])
    # Step activation for decoder output
    opt_dec_output = DouglassRachford([], [projections.bilinearMatrix])
    error_history = []
    
    print("--- Running Autoencoder Training (Step Activation) ---")
    start_time = time.time()
    train_loader = iter(train_loader)
    realData = next(train_loader)[0]
    for batch, (data,_) in enumerate(train_loader):
        inputData = np.array(realData).reshape(realData.shape[0], -1).T  # (784, batch_size)
        print(f"Processing batch {batch} with shape {inputData.shape}")
        batchError = np.zeros(inputData.shape[1])
        if batch >= num_iterations:
            break
        
        for k, sample in enumerate(inputData.T):
            # --- Forward Pass: Encoder ---
            x_in = sample
            x_in_aug = np.append(x_in, b_enc_hidden)
            x_enc_hidden = W_enc_hidden @ x_in_aug
            h_enc_hidden = step_activation(x_enc_hidden)  # Step activation
            
            h_enc_hidden_aug = np.append(h_enc_hidden, b_enc_latent)
            z_latent = W_enc_latent @ h_enc_hidden_aug
            h_latent =step_activation(z_latent)# Step activation
            
            # --- Forward Pass: Decoder ---
            h_latent_aug = np.append(h_latent, b_dec_hidden)
            x_dec_hidden = W_dec_hidden @ h_latent_aug
            h_dec_hidden = step_activation(x_dec_hidden) # relu activation
            
            h_dec_hidden_aug = np.append(h_dec_hidden, b_dec_output)
            x_reconstructed = W_dec_output @ h_dec_hidden_aug
            
            # --- Reconstruction Error ---
            reconstruction_error = (x_reconstructed - sample) ** 2
            batchError = batchError.at[k].set(np.mean(reconstruction_error))
            
            # --- Backward Pass: Update Decoder Output (with Step) ---
            w_dec_output = W_dec_output
            w_dec_hidden = W_dec_hidden
            w_enc_latent = W_enc_latent
            w_enc_hidden = W_enc_hidden
            for _ in range(5):  # Multiple projections for better convergence
                h_dec_hidden_aug, w_dec_output, x_reconstructed = opt_dec_output.step_layer(
                    h_dec_hidden_aug, w_dec_output, sample
                )
                h_dec_hidden = h_dec_hidden_aug[:-len(b_dec_output)]
                
                # --- Backward Pass: Update Decoder Hidden (bilinear) ---
                x_dec_hidden, _, h_dec_hidden = opt_dec_hidden_activation.step_layer(x_dec_hidden, np.eye(hiddenDim), h_dec_hidden)
                h_latent_aug, w_dec_hidden, x_dec_hidden = opt_dec_hidden.step_layer(h_latent_aug, w_dec_hidden, x_dec_hidden)
                h_latent = h_latent_aug[:-len(b_dec_hidden)]
                
                # --- Backward Pass: Update Encoder Latent (with Step) ---
                z_latent, _, h_latent = opt_enc_latent_activation.step_layer(z_latent, np.eye(latentDim), h_latent)
                h_enc_hidden_aug, w_enc_latent, z_latent = opt_enc_latent.step_layer(h_enc_hidden_aug, w_enc_latent, z_latent)
                h_enc_hidden = h_enc_hidden_aug[:-len(b_enc_latent)]
                
                # --- Backward Pass: Update Encoder Hidden (bilinear) ---
                x_enc_hidden, _, h_enc_hidden = opt_enc_hidden_activation.step_layer(x_enc_hidden, np.eye(hiddenDim), h_enc_hidden)
                x_in_aug, w_enc_hidden, x_enc_hidden = opt_enc_hidden.step_layer(x_in_aug, w_enc_hidden, x_enc_hidden)
            
            # --- Update Weights with learning rate ---
            lr = 1.0 / (k + 1)
            W_enc_hidden = W_enc_hidden + lr * (w_enc_hidden - W_enc_hidden)
            b_enc_hidden = b_enc_hidden + lr * (x_in_aug[-len(b_enc_hidden):] - b_enc_hidden)
            W_enc_latent = W_enc_latent + lr * (w_enc_latent - W_enc_latent)
            b_enc_latent = b_enc_latent + lr * (h_enc_hidden_aug[-len(b_enc_latent):] - b_enc_latent)
            W_dec_hidden = W_dec_hidden + lr * (w_dec_hidden - W_dec_hidden)
            b_dec_hidden = b_dec_hidden + lr * (h_latent_aug[-len(b_dec_hidden):] - b_dec_hidden)
            W_dec_output = W_dec_output + lr * (w_dec_output - W_dec_output)
            b_dec_output = b_dec_output + lr * (h_dec_hidden_aug[-len(b_dec_output):] - b_dec_output)
            
            
        
        mean_iter_error = np.mean(batchError)
        error_history.append({
            "batch": batch,
            "error": mean_iter_error.item()
        })
        
        print(f"batch {batch:4d}, Reconstruction Error: {mean_iter_error:.6f}")
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Final Error: {error_history[-1]['error']:.6f}, Time: {total_time:.2f}s\n")
    problem_params.update({
        "W_enc_hidden": W_enc_hidden, "b_enc_hidden": b_enc_hidden,
        "W_enc_latent": W_enc_latent, "b_enc_latent": b_enc_latent,
        "W_dec_hidden": W_dec_hidden, "b_dec_hidden": b_dec_hidden,
        "W_dec_output": W_dec_output, "b_dec_output": b_dec_output,
    })
    return error_history, realData


if __name__ == "__main__":
    # Initialize MNIST data module
    data_module = MNISTDataModule(batch_size=20, normalize=True)
    
    rand_key = random.PRNGKey(42)
    problem_params = setup_autoencoder_problem(data_module, rand_key)
    
    history, data = run_autoencoder_experiment(problem_params, num_iterations=100)
    
    df_results = pd.DataFrame(history)
    
    # --- Visualization: Convergence Plot ---
    plt.figure(figsize=(12, 6))
    plt.plot(df_results["batch"], df_results["error"], linewidth=2, marker='o', markersize=4)
    plt.title("Autoencoder Reconstruction Error Convergence (Step Activation)", fontsize=14)
    plt.xlabel("batch")
    plt.ylabel("Mean Reconstruction Error (MSE)")
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # --- Visualization: Reconstructed Images ---
    print("\nGenerating reconstruction visualizations...")
    
    # Get test batch
    test_images = np.array(data[:16])  # Get first 16 images
    
    # Flatten for network input
    test_images_flat = test_images.reshape(test_images.shape[0], -1)  # (16, 784)
    
    # Forward pass through autoencoder
    W_enc_hidden = problem_params["W_enc_hidden"]
    b_enc_hidden = problem_params["b_enc_hidden"]
    W_enc_latent = problem_params["W_enc_latent"]
    b_enc_latent = problem_params["b_enc_latent"]
    W_dec_hidden = problem_params["W_dec_hidden"]
    b_dec_hidden = problem_params["b_dec_hidden"]
    W_dec_output = problem_params["W_dec_output"]
    b_dec_output = problem_params["b_dec_output"]
    
    reconstructed = []
    for sample in test_images_flat:
        # Encoder
        x_in_aug = np.append(sample, b_enc_hidden)
        x_enc_hidden = W_enc_hidden @ x_in_aug
        h_enc_hidden = step_activation(x_enc_hidden)
        
        h_enc_hidden_aug = np.append(h_enc_hidden, b_enc_latent)
        z_latent = W_enc_latent @ h_enc_hidden_aug
        h_latent = step_activation(z_latent)
        
        # Decoder
        h_latent_aug = np.append(h_latent, b_dec_hidden)
        x_dec_hidden = W_dec_hidden @ h_latent_aug
        h_dec_hidden = step_activation(x_dec_hidden)
        
        h_dec_hidden_aug = np.append(h_dec_hidden, b_dec_output)
        x_reconstructed = W_dec_output @ h_dec_hidden_aug
        x_reconstructed = np.clip(x_reconstructed, 0, 1)
        
        reconstructed.append(x_reconstructed)
    
    reconstructed = np.array(reconstructed)
    
    # Plot original vs reconstructed
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    for i in range(16):
        # Original images
        axes[i // 4, (i % 4) * 2].imshow(test_images[i].reshape(28, 28), cmap='gray')
        axes[i // 4, (i % 4) * 2].set_title(f'Original ', fontsize=10)
        axes[i // 4, (i % 4) * 2].axis('off')
        
        # Reconstructed images
        axes[i // 4, (i % 4) * 2 + 1].imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        axes[i // 4, (i % 4) * 2 + 1].set_title('Reconstructed', fontsize=10)
        axes[i // 4, (i % 4) * 2 + 1].axis('off')
    
    plt.suptitle('Autoencoder: Original vs Reconstructed MNIST Digits', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # --- Visualization: Latent Space ---
    print("\nGenerating latent space visualization...")
    
    # Collect encodings and labels from training data
    all_encodings = []
    all_labels = []
    
    train_loader = iter(problem_params["train_loader"])
    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        if batch_idx >= 5:  # Use 5 batches for visualization
            break
        
        batch_x_flat = np.array(data).reshape(batch_x.shape[0], -1)
        batch_encodings = []
        
        # Encode each sample in the batch
        for sample in batch_x_flat:
            # Encoder forward pass
            x_in_aug = np.append(sample, b_enc_hidden)
            x_enc_hidden = W_enc_hidden @ x_in_aug
            h_enc_hidden = step_activation(x_enc_hidden)
            
            h_enc_hidden_aug = np.append(h_enc_hidden, b_enc_latent)
            z_latent = W_enc_latent @ h_enc_hidden_aug
            h_latent = step_activation(z_latent)
            
            batch_encodings.append(h_latent)
        
        all_encodings.append(np.array(batch_encodings))
        all_labels.append(np.array(batch_y))
    
    encodings = np.concatenate(all_encodings, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    # Apply PCA for 2D visualization
    pca = PCA(n_components=2)
    encodings_2d = pca.fit_transform(encodings)
    
    # Plot latent space
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(encodings_2d[:, 0], encodings_2d[:, 1], 
                         c=labels, cmap='tab10', s=30, alpha=0.6)
    plt.colorbar(scatter, label='Digit Class')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.title('Autoencoder Latent Space (PCA projection)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
