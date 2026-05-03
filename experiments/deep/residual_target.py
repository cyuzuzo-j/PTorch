import sys
import os
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure frameworks can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))

from frameworks.ptorch.nn.modules import Linear, ReLU
from frameworks.ptorch.optim_static import ProjectionMuon
from frameworks.ptorch.core.ops import MSEProjection
import ptorch.config as ptorch_config

sns.set_theme(style="whitegrid", context="paper", font_scale=2.0)

# ─── Configuration ────────────────────────────────────────────────────────────
D = 8               # Dimensionality of input (d) and width (W)
N_STEPS = 1000        # Total training steps (N)
N_SEEDS = 3           # Number of random seeds (K)
DEPTHS = [2, 4, 8, 16]
BATCH_SIZE = 16
LR = 0.1              # Step size (eta)

# ─── Model Definition ─────────────────────────────────────────────────────────

class ResidualMLP(nn.Module):
    """
    Residual MLP used to approximate the identity function.
    Hooks are registered on layer outputs to measure the target signal magnitude -> || A_proj - A_old ||_2.
    """
    def __init__(self, depth, d=D):
        super().__init__()
        self.depth = depth
        self.d = d
        self.layers = nn.ModuleList()
        # Build layers: Depth L means L Linear layers.
        for i in range(depth):
            self.layers.append(Linear(d, d, bias=True))
            if i < depth - 1:
                self.layers.append(ReLU())
        
        self.captured_deltas = {}  # Store per-layer delta norms
        self._setup_hooks()

    def _setup_hooks(self):
        """Attaches backward hooks to capture the target signal magnitude || h_bar - h ||_2."""
        linear_idx = 0
        for layer in self.layers:
            if isinstance(layer, Linear):
                idx = linear_idx
                def make_fwd_hook(layer_idx):
                    def hook(module, inp, out):
                        a_old = out.detach().clone()
                        def grad_hook(grad_or_target):
                            # In ptorch with projections, the "grad" flowing back is actually the projected target
                            if ptorch_config.config.use_projections:
                                delta_norm = (grad_or_target - a_old).norm(p=2).item() / a_old.size(0)
                            else:
                                delta_norm = grad_or_target.norm(p=2).item() / a_old.size(0)
                            self.captured_deltas[layer_idx] = delta_norm
                        if out.requires_grad:
                            out.register_hook(grad_hook)
                    return hook
                
                layer.register_forward_hook(make_fwd_hook(idx))
                linear_idx += 1

    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, Linear):
                x = x + layer(x)
            else:
                x = layer(x)
        return x


# ─── Experiment Runner ────────────────────────────────────────────────────────

def run_experiment(device):
    results_deltas = {}       # {depth: {step: [delta_layer_0, ..., delta_layer_L-1]}}
    results_final_loss = {}   # {depth: [loss_seed_1, ..., loss_seed_K]}
    
    # We will record signals at these steps (1-indexed)
    record_steps = [1, N_STEPS // 2, N_STEPS]

    for depth in DEPTHS:
        print(f"\n--- Running Depth L={depth} ---")
        results_deltas[depth] = {t: [] for t in record_steps}
        results_final_loss[depth] = []
        
        for seed in range(N_SEEDS):
            torch.manual_seed(seed)
            random.seed(seed)
            
            model = ResidualMLP(depth=depth, d=D).to(device)
            optimizer = ProjectionMuon(model.parameters()) # Using standard default params
            
            # Dataset: Gaussian inputs
            # Instead of static dataset, generate batch dynamically at each step
            final_loss = 0.0
            
            with ptorch_config.config.projections(True):
                for step in range(1, N_STEPS + 1):
                    x_batch = torch.randn(BATCH_SIZE, D, device=device)
                    y_batch = x_batch.clone()
                    
                    model.train()
                    optimizer.zero_grad()
                    
                    preds = model(x_batch)
                    
                    # MSE projection operation
                    loss = MSEProjection.apply(preds, y_batch)
                    loss.backward()
                    optimizer.step()
                    
                    if step in record_steps:
                        # Extract deltas, sorted by layer index (0 to depth-1)
                        # We reverse it for plotting (0 = output layer, depth-1 = input side)
                        # Actually, wait. The text says "counting backward from the output".
                        # Let's collect them from input to output natively:
                        deltas = [model.captured_deltas.get(k, 0.0) for k in range(depth)]
                        results_deltas[depth][step].append(deltas)
                        
                    if step == N_STEPS:
                        final_loss = loss.item() / BATCH_SIZE
            
            results_final_loss[depth].append(final_loss)
            print(f"  Seed {seed+1}/{N_SEEDS} | Final Loss: {final_loss:.6f}")

    return results_deltas, results_final_loss

# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_results(results_deltas, results_final_loss, out_dir="figures"):
    os.makedirs(out_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Figure A: Target Signal Profile ---
    # We'll plot for t = 1 (Initial step) to see the pure backward decay
    t_plot = 1
    colors = sns.color_palette("viridis", len(DEPTHS))
    
    for i, depth in enumerate(DEPTHS):
        # results_deltas[depth][t_plot] is a list of lists: shape (N_SEEDS, depth)
        data = torch.tensor(results_deltas[depth][t_plot]) # (K, L)
        mean_deltas = data.mean(dim=0).tolist()
        std_deltas = data.std(dim=0).tolist()
        
        # x-axis: distance from output (1 = output, L = input layer)
        x_axis = list(range(1, depth + 1))
        
        ax1.plot(x_axis, mean_deltas, marker='o', color=colors[i], label=f"L={depth}")
        ax1.fill_between(x_axis, [m - s for m, s in zip(mean_deltas, std_deltas)], [m + s for m, s in zip(mean_deltas, std_deltas)], color=colors[i], alpha=0.2)
        
    ax1.set_xlabel("Distance from Output Layer")
    ax1.set_ylabel(r"Target Signal Magnitude $\delta_k^{(t)}$")
    ax1.set_yscale("log")
    ax1.set_title(r"(a) Residual Signal Profile (Step 1)")
    ax1.legend(loc="lower left", fontsize=12)
    ax2.set_xscale("log", base=2)

    ax1.invert_xaxis() # Make it decay from left to right as we go deeper
    
    # --- Figure B: Final Loss vs Depth ---
    mean_losses = [torch.tensor(results_final_loss[d]).mean().item() for d in DEPTHS]
    std_losses = [torch.tensor(results_final_loss[d]).std().item() for d in DEPTHS]
    
    ax2.plot(DEPTHS, mean_losses, marker='s', color='darkred', linewidth=2)
    ax2.fill_between(DEPTHS, 
                     [m - s for m, s in zip(mean_losses, std_losses)], 
                     [m + s for m, s in zip(mean_losses, std_losses)], 
                     color='red', alpha=0.2)
    
    ax2.set_xlabel("Network Depth $L$")
    ax2.set_ylabel(r"Final Training Loss (MSE)")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(DEPTHS)
    ax2.set_xticklabels(DEPTHS)
    ax2.set_title(r"(b) Final Loss vs. Depth $L$")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "residual_combined.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "residual_combined.png"), dpi=300, bbox_inches="tight")
    print(f"\nSaved figures to {out_dir}/")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    deltas, final_losses = run_experiment(device)
    plot_results(deltas, final_losses)

