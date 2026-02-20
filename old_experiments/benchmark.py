# benchmark.py
import jax
import jax.numpy as np
import jax.random as random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import tracemalloc
import torch
import torch.nn as nn
import torch.optim as optim

import tools.projections as projections
from tools.optimize import DouglassRachford , AlternatingProjection

def he_initialization(key, shape, fan_in):
    """He initialization: samples from N(0, sqrt(2/fan_in))."""
    std = np.sqrt(2.0 / fan_in)
    return random.normal(key, shape) * std

def setup_problem(rand_key):
    """Initializes the neural network problem parameters and data."""
    keys = random.split(rand_key, 6)
    
    # --- Hyperparameters ---
    inputDim = 10
    hiddenDim = 128
    outputDim = 1
    samples = 2**inputDim
    delta = 1.0

    # --- Data Generation (shapes match original code) ---
    inputData = random.choice(keys[0], np.array([0, 1]), shape=(inputDim, samples))
    print("Input Data Shape:", inputData.shape)
    outputData = random.choice(keys[1], np.array([0, 1]), shape=(outputDim, samples))

    # --- Weight and Bias Initialization ---
    W0 = he_initialization(keys[2], (hiddenDim, inputDim), inputDim)
    W0 = np.concatenate((W0, np.eye(hiddenDim)), axis=1) 
    W1 = he_initialization(keys[3], (outputDim, hiddenDim),hiddenDim )
    W1 = np.concatenate((W1, np.eye(outputDim)), axis=1) 

    b0 = random.normal(keys[4],(hiddenDim,))
    
    b1 = random.normal(keys[5],(outputDim,))

    problem_params = {
        "W0": W0, "b0": b0, "W1": W1, "b1": b1,
        "inputData": inputData, "outputData": outputData,
        "delta": delta, "hiddenDim": hiddenDim,
    }
    return problem_params


# %%
# =============================================================================
# BENCHMARKING CONFIGURATION
# =============================================================================
# Define all experiments to run. Add new configurations here.
"""
CONFIGURATIONS = [
    {
        "name": "DR + Step Activation",
        "optimizer": DouglassRachford,
        "activation": projections.stepActivation,
        "extraConstraints": [projections.orthonormalizeMatrix],
    },
    {
        "name": "DR + Step Activation (no normalization)",
        "optimizer": DouglassRachford,
        "activation": projections.stepActivation,
        "extraConstraints": [],
10
    },
    {
        "name": "DR + ReLU Activation",
        "optimizer": DouglassRachford,
        "activation": projections.sum_relu_proj,
        "extraConstraints": [projections.orthonormalizeMatrix],
    },
    {
        "name": "Dr + ReLU Activation (no normalization)",
        "optimizer": DouglassRachford,
        "activation": projections.sum_relu_proj,
        "extraConstraints": [],
    },
]
"""
CONFIGURATIONS = [
    {
        "name": "DR + Step Activation",
        "optimizer": DouglassRachford,
        "activation": projections.stepActivation,
        "extraConstraints": [],
    },
    {
        "name": "AP + Step Activation",
        "optimizer": AlternatingProjection,
        "activation": projections.stepActivation,
        "extraConstraints": [],
    },
    {
        "name": "AP + ReLU Activation",
        "optimizer": AlternatingProjection,
        "activation": projections.sum_relu_proj,
        "extraConstraints": [],
    },
    {
        "name": "DR + ReLU Activation",
        "optimizer": DouglassRachford,
        "activation": projections.sum_relu_proj,
        "extraConstraints": [],
    }


]

# %%
# =============================================================================
# Experiment Runner (Preserves Original Logic)
# =============================================================================

def run_experiment(config, problem_params, num_iterations=1000):
    """Runs a single experiment, preserving the original update logic."""
    
    # --- Unpack problem parameters ---
    W0, b0, W1, b1 = problem_params["W0"], problem_params["b0"], problem_params["W1"], problem_params["b1"]
    inputData, outputData = problem_params["inputData"], problem_params["outputData"]
    delta = problem_params["delta"]
    activation = config["activation"]
    extraConstraints = config["extraConstraints"]
    
    # --- Instantiate optimizers based on the config ---
    opt_class = config["optimizer"]
    
    optInput = opt_class(extraConstraints, [projections.bilinearMatrix])
    optActivation = opt_class([], [activation])
    optHidden = opt_class([], [projections.bilinearMatrix])
    optOutput = opt_class([lambda x,w,y :projections.classifierOutputVector(x,w,y,delta=delta)], [])
    
    error_history = []
    
    print(f"--- Running: {config['name']} ---")
    tracemalloc.start()
    start_time = time.time()
    
    # THE CORE LOOP: This section is structured to be functionally identical
    # to your original code snippet to avoid breaking sensitive logic.
    for iteration in range(num_iterations):
        batchError = np.zeros(len(outputData))
        for k, sample in enumerate(outputData.T):
            ## do a forward pass
            x_in = inputData[:,k]
            x_in_aug = np.append(x_in, b0)
            x_hidden = W0 @ x_in_aug
            
            if activation == projections.stepActivation:
                h_hidden = np.where(x_hidden >= 0, 1.0, 0.0)  # step activation
            else:  # relu
                h_hidden = x_hidden * (x_hidden > 0)  # ReLU activation
            
            h_hidden_aug = np.append(h_hidden, b1)
            x_out = W1 @ h_hidden_aug
            
            ## calculate error
            for i, classSubSample in enumerate(sample):
                if classSubSample==1:
                    batchError = batchError.at[k].set(batchError[k] + (x_out[i] - delta)**2 if x_out[i] < delta else 0)
                else:
                    batchError = batchError.at[k].set(batchError[k] + (x_out[i])**2 if x_out[i] > 0 else 0)

            x_out, _, _ = optOutput.step_layer(x_out, np.eye(sample.shape[0]), sample)
            h_hidden_aug, w1_proj, x_out = optHidden.step_layer(h_hidden_aug, W1, x_out)
            h_hidden= h_hidden_aug[:][:-1]

            x_hidden, _, h_hidden = optActivation.step_layer(x_hidden, np.eye(len(x_hidden)), h_hidden)
            
            x_in_aug, w0_proj, x_hidden = optInput.step_layer(x_in_aug, W0, x_hidden)

            b1 = b1 + (1/(k+1))*(h_hidden_aug[:][-1:] - b1)
            b0 = b0 + (1/(k+1))*(x_in_aug[:][-len(b0):] - b0) 
            
            W0 = W0 + (1/(k+1))*(w0_proj - W0)
            W1 = W1 + (1/(k+1))*(w1_proj - W1)
        mean_iter_error = np.mean(batchError)
        
        error_history.append({
            "config_name": config["name"],
            "iteration": iteration,
            "error": mean_iter_error.item()
        })
        if iteration % 100 == 0:
            print(f"Iter {iteration:4d}, Avg Error: {mean_iter_error:.6f}")

    end_time = time.time()
    
    # --- Memory Profiling ---
    try:
        # Save device memory profile
        sanitized_name = config['name'].replace(' ', '_').replace('+', '').replace('(', '').replace(')', '')
        profile_path = f"memory_{sanitized_name}.prof"
        jax.profiler.save_device_memory_profile(profile_path)
        print(f"Saved device memory profile to {profile_path}")
        
        # Print device memory stats if available
        devices = jax.devices()
        if devices:
            dev = devices[0]
            if hasattr(dev, "memory_stats"):
                stats = dev.memory_stats()
                if stats:
                    print(f"Device Memory Stats: {stats}")
    except Exception as e:
        print(f"Memory profiling note: {e}")

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_time = end_time - start_time
    print(f"Final Error: {error_history[-1]['error']:.6f}, Time: {total_time:.2f}s, Peak Memory: {peak / 10**6:.2f}MB\n")
    
    for record in error_history:
        record['time_s'] = total_time
        record['peak_memory_mb'] = peak / 10**6

    return error_history

def run_pytorch_experiment(problem_params, num_iterations=1000, learning_rate=0.01):
    """Runs a PyTorch ADAM experiment for comparison."""
    
    # --- Unpack and Convert to PyTorch ---
    # Transpose inputData to (samples, inputDim) for PyTorch standard [N, C]
    inputData = torch.tensor(np.array(problem_params["inputData"]).T, dtype=torch.float32)
    outputData = torch.tensor(np.array(problem_params["outputData"]).T, dtype=torch.float32)
    
    inputDim = problem_params["inputData"].shape[0]
    hiddenDim = problem_params["hiddenDim"]
    outputDim = problem_params["outputData"].shape[0]
    delta = problem_params["delta"]

    # --- Define Model ---
    model = nn.Sequential(
        nn.Linear(inputDim, hiddenDim),
        nn.ReLU(),
        nn.Linear(hiddenDim, outputDim)
    )
    
    # Copy weights
    with torch.no_grad():
        model[0].weight.data = torch.tensor(np.array(problem_params["W0"][:, :inputDim]), dtype=torch.float32)
        model[0].bias.data = torch.tensor(np.array(problem_params["b0"]), dtype=torch.float32)
        model[2].weight.data = torch.tensor(np.array(problem_params["W1"][:, :hiddenDim]), dtype=torch.float32)
        model[2].bias.data = torch.tensor(np.array(problem_params["b1"]), dtype=torch.float32)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    mse = nn.MSELoss()
    error_history = []
    print(f"--- Running: PyTorch ADAM (lr={learning_rate}) ---")
    tracemalloc.start()
    start_time = time.time()
    
    for iteration in range(num_iterations):
        epoch_error = 0.0
        
        # Iterate through samples to match the online nature of the benchmark
        for k in range(len(inputData)):
            x = inputData[k:k+1]
            y = outputData[k:k+1]
            
            optimizer.zero_grad()
            out = model(x)
            
            # Squared Hinge Loss
            loss = mse(out, y)
            loss_scalar = loss.item()
            loss.backward()
            optimizer.step()
            
            epoch_error += loss_scalar
            
        mean_iter_error = epoch_error / len(inputData)
        
        error_history.append({
            "config_name": f"PyTorch ADAM (lr={learning_rate})",
            "iteration": iteration,
            "error": mean_iter_error
        })
        
        if iteration % 100 == 0:
            print(f"Iter {iteration:4d}, Avg Error: {mean_iter_error:.6f}")

    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_time = end_time - start_time
    print(f"Final Error: {error_history[-1]['error']:.6f}, Time: {total_time:.2f}s, Peak Memory: {peak / 10**6:.2f}MB\n")
    
    for record in error_history:
        record['time_s'] = total_time
        record['peak_memory_mb'] = peak / 10**6

    return error_history

# %%
# =============================================================================
# Main Execution and Visualization
# =============================================================================
import random as rn
if __name__ == "__main__":
    rand_key = random.PRNGKey(rn.randint(0, 10000))
    
    # --- Warm-up ---
    print("--- Starting Warm-up ---")
    rand_key, subkey = random.split(rand_key)
    warmup_params = setup_problem(subkey)
    
    for config_item in CONFIGURATIONS:
        # Run briefly to trigger JIT compilation
        run_experiment(config_item, warmup_params, num_iterations=10)
        
    run_pytorch_experiment(warmup_params, num_iterations=10)
    print("--- Warm-up Complete ---")

    all_results = []
    
    for _ in range(50):  # Run multiple trials
        rand_key, subkey = random.split(rand_key)
        problem_params = setup_problem(subkey)
        
        # Run Projection-based configs
        for config_item in CONFIGURATIONS:
            history = run_experiment(config_item, problem_params, num_iterations=100)
            all_results.extend(history)
            
        # Run PyTorch SGD
        history_pt = run_pytorch_experiment(problem_params, num_iterations=100, learning_rate=0.01)
        all_results.extend(history_pt)

    df_results = pd.DataFrame(all_results)

    # --- Visualization ---
    sns.set_theme(style="whitegrid")
    sns.set_context("poster")  # Increase font scale for slides
    print(df_results)

    plt.figure(figsize=(12, 10))
    ax1 = sns.lineplot(data=df_results, x="iteration", y="error", hue="config_name", linewidth=3)
    ax1.set_title("Error Convergence by Optimizer", fontsize=30, pad=20)
    ax1.set_ylabel("Average Error", fontsize=24)
    ax1.set_xlabel("Iteration", fontsize=24)
    ax1.set_yscale('log')
    plt.legend(fontsize=20, title_fontsize=22)
    plt.show()

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

    plt.suptitle("Benchmark Summary", fontsize=36, y=1.05)
    plt.tight_layout()
    plt.show()
# %%
