"""
Vanishing Target Signal Experiment
===================================
Measures how the projection-based target signal δ_k = ‖A_proj − A_old‖₂
decays as it propagates backward through an L-layer linear MLP trained to
approximate the identity function (x → x) with MSE projection loss.

Produces two plots:
  (a) Target signal magnitude per layer at step 1 for each depth.
  (b) Final MSE training loss vs. network depth.
"""

import sys
import os
import random

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from ptorch.nn.modules import Linear
from ptorch.optim_static import ProjectionMuonV2
from ptorch.core.ops import MSEProjection

sns.set_theme(style="whitegrid", context="paper", font_scale=2.0)

D = 32
N_STEPS = 5000
N_SEEDS = 3
DEPTHS = [2, 4, 8]
BATCH_SIZE = 16


class IdentityMLP(nn.Module):
    """L-layer linear MLP with hooks that capture ‖A_proj − A_old‖₂ per layer."""

    def __init__(self, depth, d=D):
        super().__init__()
        self.depth = depth
        self.d = d
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(Linear(d, d, bias=True))

        self.captured_deltas = {}
        self._setup_hooks()

    def _setup_hooks(self):
        linear_idx = 0
        for layer in self.layers:
            if isinstance(layer, Linear):
                idx = linear_idx
                def make_fwd_hook(layer_idx):
                    def hook(module, inp, out):
                        a_old = out.detach().clone()
                        def target_hook(target):
                            delta_norm = (target - a_old).norm(p=2).item() / a_old.size(0)
                            self.captured_deltas[layer_idx] = delta_norm
                        if out.requires_grad:
                            out.register_hook(target_hook)
                    return hook

                layer.register_forward_hook(make_fwd_hook(idx))
                linear_idx += 1

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def run_experiment(device):
    results_deltas = {}
    results_final_loss = {}
    record_steps = [1, N_STEPS // 2, N_STEPS]

    for depth in DEPTHS:
        print(f"\n--- Running Depth L={depth} ---")
        results_deltas[depth] = {t: [] for t in record_steps}
        results_final_loss[depth] = []

        for seed in range(N_SEEDS):
            torch.manual_seed(seed)
            random.seed(seed)

            model = IdentityMLP(depth=depth, d=D).to(device)
            optimizer = ProjectionMuonV2(model.parameters())
            final_loss = 0.0

            for step in range(1, N_STEPS + 1):
                x_batch = torch.randn(BATCH_SIZE, D, device=device)
                y_batch = x_batch.clone()

                model.train()
                optimizer.zero_grad()

                preds = model(x_batch)
                loss = MSEProjection.apply(preds, y_batch)
                loss.backward()
                optimizer.step()

                if step in record_steps:
                    deltas = [model.captured_deltas.get(k, 0.0) for k in range(depth)]
                    results_deltas[depth][step].append(deltas)

                if step == N_STEPS:
                    final_loss = loss.item() / BATCH_SIZE

            results_final_loss[depth].append(final_loss)
            print(f"  Seed {seed+1}/{N_SEEDS} | Final Loss: {final_loss:.6f}")

    return results_deltas, results_final_loss


def plot_results(results_deltas, results_final_loss, out_dir="images"):
    os.makedirs(out_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    t_plot = 1
    
    colors = sns.color_palette("viridis", len(DEPTHS))

    for i, depth in enumerate(DEPTHS):
        data = torch.tensor(results_deltas[depth][t_plot])
        x_axis = range(1, depth + 1)
        mean_deltas_tensor = data.mean(dim=0)
        std_deltas_tensor = data.std(dim=0)
        ax1.plot(x_axis, mean_deltas_tensor.tolist(), marker='o', color=colors[i], label=f"L={depth}")
        ax1.fill_between(x_axis, (mean_deltas_tensor - std_deltas_tensor).tolist(), (mean_deltas_tensor + std_deltas_tensor).tolist(), color=colors[i], alpha=0.2)

    ax1.set_xlabel("Distance from Input Layer")
    ax1.set_ylabel(r"Target Signal Magnitude $\delta_k^{(t)}$")
    ax1.set_yscale("log")
    ax1.set_title(r"(a) Vanishing Signal Profile (Step 1)")
    ax1.legend(loc="lower left", fontsize=12)
    ax1.invert_xaxis()

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
    plt.savefig(os.path.join(out_dir, "vanishing_combined.pdf"), dpi=300, bbox_inches="tight")
    print(f"\nSaved figures to {out_dir}/")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    deltas, final_losses = run_experiment(device)
    plot_results(deltas, final_losses)
