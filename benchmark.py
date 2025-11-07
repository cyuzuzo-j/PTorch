# benchmark.py
import jax.numpy as np
import jax.random as random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

import projections
from optimize import DouglassRachford , AlternatingProjection


def setup_problem(rand_key):
    """Initializes the neural network problem parameters and data."""
    keys = random.split(rand_key, 4)
    
    # --- Hyperparameters ---
    inputDim = 50
    hiddenDim = 50
    outputDim = 2
    samples = 50
    delta = 1.0

    # --- Data Generation (shapes match original code) ---
    inputData = random.choice(keys[0], np.array([0, 1]), shape=(inputDim, samples))
    outputData = random.choice(keys[1], np.array([0, 1]), shape=(outputDim, samples))

    # --- Weight and Bias Initialization ---
    W0 = random.normal(keys[2], (hiddenDim, inputDim))
    W0 = np.concatenate((W0, np.eye(hiddenDim)), axis=1) 
    W1 = random.normal(keys[3], (outputDim, hiddenDim))
    W1 = np.concatenate((W1, np.eye(outputDim)), axis=1) 

    b0 = random.normal(rand_key,(hiddenDim,))
    b1 = random.normal(rand_key,(outputDim,))

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
        "name": "DR + Step Activation ( orthonormalization)",
        "optimizer": AlternatingProjection,
        "activation": projections.stepActivation,
        "extraConstraints": [projections.orthonormalizeMatrix],

    },
    {
        "name": "DR + Step Activation ",
        "optimizer": AlternatingProjection,
        "activation": projections.stepActivation,
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
    total_time = end_time - start_time
    print(f"Final Error: {error_history[-1]['error']:.6f}, Time: {total_time:.2f}s\n")
    
    for record in error_history:
        record['time_s'] = total_time

    return error_history

# %%
# =============================================================================
# Main Execution and Visualization
# =============================================================================
import random as rn
if __name__ == "__main__":
    rand_key = random.PRNGKey(rn.randint(0, 10000))
    all_results = []
    
    for config_item in CONFIGURATIONS:
        rand_key, subkey = random.split(rand_key)
        problem_params = setup_problem(subkey)
        
        history = run_experiment(config_item, problem_params, num_iterations=500)
        all_results.extend(history)

    df_results = pd.DataFrame(all_results)

    # --- Visualization ---
    sns.set_theme(style="whitegrid")
    print(df_results)

    plt.figure(figsize=(12, 7))
    ax1 = sns.lineplot(data=df_results, x="iteration", y="error", hue="config_name")
    ax1.set_title("Error Convergence by Optimizer", fontsize=16)
    ax1.set_ylabel("Average Error")
    ax1.set_yscale('log')
    plt.show()

    df_summary = df_results.loc[df_results.groupby('config_name')['iteration'].idxmax()]
    
    fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(15, 6))
    sns.barplot(data=df_summary, x="error", y="config_name", ax=ax2, palette="viridis")
    ax2.set_title("Final Error (Lower is Better)")
    ax2.set_xlabel("Final Average Error")
    ax2.set_ylabel("")

    sns.barplot(data=df_summary, x="time_s", y="config_name", ax=ax3, palette="plasma")
    ax3.set_title("Total Runtime (Lower is Better)")
    ax3.set_xlabel("Time (seconds)")
    ax3.set_ylabel("")

    plt.suptitle("Benchmark Summary", fontsize=18)
    plt.tight_layout()
    plt.show()
# %%
