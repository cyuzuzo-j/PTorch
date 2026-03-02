import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))

import jax
import jax.numpy as jnp
import pjax
from pjax import nn, optim
from experiments.shared.data import MNISTDataModule
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
import tracemalloc
import torch
import torch.nn as tnn
import torch.optim as toptim
import numpy as np

# 1. Define the PJAX autoencoder model
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

def run_pjax_experiment(config, num_epochs=5):
    print(f"--- Running: {config['name']} ---")
    
    # Initialize data
    dataset = MNISTDataModule(batch_size=config['batch_size'])
    train_data = dataset.train_dataloader()
    
    # Initialize model
    key = jax.random.key(config.get('seed', 0))
    model = Autoencoder(in_features=784, hidden_features=256, latent_features=16)
    params = model.init(key)
    
    # Initialize optimizer
    optimizer_class = config['optimizer']
    optimizer = optimizer_class(**config['optimizer_params'])
    
    @jax.jit
    def train_step(params, x):
        def apply_fn(params):
            reconstruction = model.apply(params, x)
            return pjax.means_squared_error(reconstruction, x)    
        
        updated_params, loss = optimizer.update(apply_fn, params)
        return updated_params, loss

    error_history = []
    tracemalloc.start()
    start_time = time.time()
    
    global_step = 0
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        count = 0
        for i, (x, y) in enumerate(train_data):
            x = jnp.array(x) # Ensure JAX array
            x = x.reshape(x.shape[0], -1)  # flatten
            params, loss = train_step(params, x)
            
            loss_val = loss.item()
            epoch_loss += loss_val
            count += 1
            
            # Record metrics periodically
            if i % 10 == 0:
                 error_history.append({
                    "config_name": config["name"],
                    "iteration": global_step,
                    "error": loss_val,
                    "epoch": epoch
                })
            global_step += 1
            
        avg_epoch_loss = epoch_loss / count
        print(f"Epoch {epoch}, Avg Loss: {avg_epoch_loss:.4f}")

    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_time = end_time - start_time
    
    print(f"Final Error: {error_history[-1]['error']:.6f}, Time: {total_time:.2f}s, Peak Memory: {peak / 10**6:.2f}MB\n")
    
    # Add summary stats to all records
    for record in error_history:
        record['time_s'] = total_time
        record['peak_memory_mb'] = peak / 10**6
        
    return error_history

# PyTorch Implementation
class PyTorchAutoencoder(tnn.Module):
    def __init__(self, in_features, hidden_features, latent_features):
        super().__init__()
        self.encoder = tnn.Sequential(
            tnn.Linear(in_features, hidden_features),
            tnn.ReLU(),
            tnn.Linear(hidden_features, latent_features)
        )
        self.decoder = tnn.Sequential(
            tnn.Linear(latent_features, hidden_features),
            tnn.ReLU(),
            tnn.Linear(hidden_features, in_features)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def run_pytorch_experiment(config, num_epochs=5):
    print(f"--- Running: {config['name']} ---")
    
    dataset = MNISTDataModule(batch_size=config['batch_size'])
    train_data = dataset.train_dataloader()
    
    model = PyTorchAutoencoder(in_features=784, hidden_features=256, latent_features=16)
    optimizer = toptim.Adam(model.parameters(), lr=config.get('lr', 1e-3))
    criterion = tnn.MSELoss()
    
    error_history = []
    tracemalloc.start()
    start_time = time.time()
    
    global_step = 0
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        count = 0
        for i, (x, y) in enumerate(train_data):
            x = x.view(x.size(0), -1) # Flatten
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, x)
            loss.backward()
            optimizer.step()
            
            loss_val = loss.item()
            epoch_loss += loss_val
            count += 1
            
            if i % 10 == 0:
                error_history.append({
                    "config_name": config["name"],
                    "iteration": global_step,
                    "error": loss_val,
                    "epoch": epoch
                })
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss_val:.6f}")
            global_step += 1
            
        avg_epoch_loss = epoch_loss / count
        print(f"Epoch {epoch}, Avg Loss: {avg_epoch_loss:.4f}")

    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_time = end_time - start_time
    
    print(f"Final Error: {error_history[-1]['error']:.6f}, Time: {total_time:.2f}s, Peak Memory: {peak / 10**6:.2f}MB\n")
    
    for record in error_history:
        record['time_s'] = total_time
        record['peak_memory_mb'] = peak / 10**6
        
    return error_history

if __name__ == "__main__":
    # Configurations
    configs = [
        {
            "name": "PJAX DouglasRachford",
            "type": "pjax",
            "optimizer": optim.DouglasRachford,
            "optimizer_params": {"steps_per_update": 1},
            "batch_size": 16
        },
        {
            "name": "PyTorch Adam",
            "type": "pytorch",
            "lr": 1e-3,
            "batch_size": 16
        }
    ]
    
    all_results = []
    
    for config in configs:
        if config["type"] == "pjax":
            history = run_pjax_experiment(config, num_epochs=2) # Reduced epochs for speed in demo
        else:
            history = run_pytorch_experiment(config, num_epochs=2)
        all_results.extend(history)
        
    df_results = pd.DataFrame(all_results)
    
    # Visualization (same as benchmark.py)
    sns.set_theme(style="whitegrid")
    sns.set_context("poster")
    
    plt.figure(figsize=(12, 10))
    ax1 = sns.lineplot(data=df_results, x="iteration", y="error", hue="config_name", linewidth=3)
    ax1.set_title("Autoencoder Training Convergence", fontsize=30, pad=20)
    ax1.set_ylabel("MSE Loss", fontsize=24)
    ax1.set_xlabel("Iteration (Batch)", fontsize=24)
    ax1.set_yscale('log')
    plt.legend(fontsize=20, title_fontsize=22)
    plt.show()
    
    # Summary plots
    df_summary = df_results.loc[df_results.groupby('config_name')['iteration'].idxmax()]
    
    fig, (ax2, ax3, ax4) = plt.subplots(1, 3, figsize=(30, 8))
    sns.barplot(data=df_summary, x="error", y="config_name", ax=ax2, palette="viridis")
    ax2.set_title("Final Error (Lower is Better)", fontsize=28, pad=15)
    ax2.set_xlabel("Final Average Error", fontsize=24)
    ax2.set_ylabel("", fontsize=24)
    ax2.tick_params(axis='both', labelsize=20)
    
    sns.barplot(data=df_summary, x="time_s", y="config_name", ax=ax3, palette="plasma")
    ax3.set_title("Total Runtime (Lower is Better)", fontsize=28, pad=15)
    ax3.set_xlabel("Time (seconds)", fontsize=24)
    ax3.set_ylabel("", fontsize=24)
    ax3.tick_params(axis='both', labelsize=20)
    
    sns.barplot(data=df_summary, x="peak_memory_mb", y="config_name", ax=ax4, palette="magma")
    ax4.set_title("Peak Memory Usage (Lower is Better)", fontsize=28, pad=15)
    ax4.set_xlabel("Memory (MB)", fontsize=24)
    ax4.set_ylabel("", fontsize=24)
    ax4.tick_params(axis='both', labelsize=20)
    
    plt.suptitle("Autoencoder Benchmark Summary", fontsize=36, y=1.05)
    plt.tight_layout()
    plt.show()