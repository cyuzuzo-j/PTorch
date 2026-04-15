import sys
import os
import argparse
import random
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure frameworks can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))

from frameworks.ptorch.nn.modules import LinearHybrid, ReLUHybrid, _parse_norm
from frameworks.ptorch.optim_static import ProjectionSGD, ProjectionAdam, ProjectionAdadelta, ProjectionMuon
from frameworks.ptorch.core.ops import CrossEntropyProjection
import ptorch.config as ptorch_config
from experiments.shared.data import MNISTDataModule

# Styling similar to vanishing targets target 
sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)

# ─── Configuration ────────────────────────────────────────────────────────────
EPOCHS = 10                  # Total epochs
BATCH_SIZE = 64
SEEDS = [42]                 # Use fewer seeds for fast grid search, or [42, 43, 44] if needed
CONVERGENCE_THRESH = 0.05    # Threshold for convergence epoch
NORM = "inf"                 # default ptorch norm for these experiments

# Step sizes to sweep
LRS = [0.01, 0.1, 0.5, 1.0, 2.0]

# ─── Model Definition ─────────────────────────────────────────────────────────

class PtorchMLP(nn.Module):
    """
    MLP used for evaluating optimizers on Ptorch.
    """
    def __init__(self, in_dim=784, hidden_dims=[128, 128], out_dim=10, norm="inf"):
        super().__init__()
        self.layers = nn.ModuleList()
        last = in_dim
        
        # Build hidden layers
        for d in hidden_dims:
            self.layers.append(LinearHybrid(last, d, norm=norm))
            self.layers.append(ReLUHybrid(norm=norm))
            last = d
        
        # Output layer
        self.output_layer = LinearHybrid(last, out_dim, norm=norm)
        
    def forward(self, x):
        # Flatten x
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)

# ─── Experiment Runner ────────────────────────────────────────────────────────

def train_and_eval(opt_cls, opt_kwargs, lr, device, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Dataset
    dataset = MNISTDataModule(batch_size=BATCH_SIZE, seed=seed)
    train_loader = dataset.train_dataloader()
    test_loader = dataset.test_dataloader()
    
    model = PtorchMLP(norm=NORM).to(device)
    
    # Init optimizer
    # ProjectionSGD, ProjectionAdam, etc.
    optimizer_cls = opt_cls
    if optimizer_cls == ProjectionSGD:
        optimizer = optimizer_cls(model.parameters(), lr=lr, momentum=0.9, **opt_kwargs)
    else:
        optimizer = optimizer_cls(model.parameters(), lr=lr, **opt_kwargs)
    
    epoch_losses = []
    convergence_epoch = -1
    
    with ptorch_config.config.projections(True):
        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss = 0.0
            num_batches = 0
            
            for x_batch, y_batch in train_loader:
                x_batch = torch.tensor(x_batch).float().to(device)
                y_batch = torch.tensor(y_batch).long().to(device)
                
                optimizer.zero_grad()
                preds = model(x_batch)
                
                import torch.nn.functional as F
                y_oh = F.one_hot(y_batch, num_classes=preds.shape[-1]).float()
                # Cross entropy projection
                loss = CrossEntropyProjection.apply(preds, y_oh)
                loss_sum = loss.sum()
                loss_sum.backward()
                optimizer.step()
                
                # compute conventional loss for recording
                ce_loss = F.cross_entropy(preds.detach(), y_batch)
                total_loss += ce_loss.item()
                num_batches += 1
                
            avg_loss = total_loss / num_batches
            epoch_losses.append(avg_loss)
            
            if convergence_epoch == -1 and avg_loss < CONVERGENCE_THRESH:
                convergence_epoch = epoch
                
    # Evaluate accuracy
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        with ptorch_config.config.projections(False): # Usually eval without projection overhead, but standard forward
            for x_batch, y_batch in test_loader:
                x_batch = torch.tensor(x_batch).float().to(device)
                y_batch = torch.tensor(y_batch).long().to(device)
                
                preds = model(x_batch)
                _, predicted = torch.max(preds.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
                
    test_acc = 100.0 * correct / total
    
    return epoch_losses, test_acc, convergence_epoch

def run_grid_search(device):
    optimizers_to_test = {
        "SGD (momentum)": (ProjectionSGD, {}),
        "Adam": (ProjectionAdam, {"betas": (0.9, 0.999)}),
        "Adadelta": (ProjectionAdadelta, {"rho": 0.9}),
        "Muon": (ProjectionMuon, {})
    }
    
    best_results = {}
    
    for opt_name, (opt_cls, opt_kwargs) in optimizers_to_test.items():
        print(f"\n--- Grid Search for {opt_name} ---")
        best_lr = None
        best_loss_curve = None
        best_acc = -1
        best_conv = -1
        lowest_final_loss = float('inf')
        
        for lr in LRS:
            print(f"  Testing LR = {lr}...", end="", flush=True)
            accs = []
            convs = []
            all_curves = []
            
            try:
                for seed in SEEDS:
                    losses, acc, conv = train_and_eval(opt_cls, opt_kwargs, lr, device, seed)
                    accs.append(acc)
                    convs.append(conv if conv != -1 else EPOCHS + 1)
                    all_curves.append(losses)
                
                avg_acc = sum(accs) / len(SEEDS)
                avg_conv = sum(convs) / len(SEEDS)
                
                # Average loss curve
                avg_curve = [sum(col) / len(SEEDS) for col in zip(*all_curves)]
                final_loss = avg_curve[-1]
                
                print(f" Final Loss: {final_loss:.4f} | Acc: {avg_acc:.2f}% | Conv: {avg_conv:.1f}")
                
                if final_loss < lowest_final_loss:
                    lowest_final_loss = final_loss
                    best_lr = lr
                    best_loss_curve = avg_curve
                    best_acc = avg_acc
                    best_conv = avg_conv
            except Exception as e:
                print(f" Failed! ({e})")
                
        best_results[opt_name] = {
            "lr": best_lr,
            "curve": best_loss_curve,
            "acc": best_acc,
            "conv": best_conv if best_conv <= EPOCHS else None
        }
        
    return best_results

# ─── Plotting & Output ────────────────────────────────────────────────────────

def plot_and_summarize(results, out_dir="figures"):
    os.makedirs(out_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = sns.color_palette("deep", len(results))
    
    print("\n\n" + "="*60)
    print("Optimization Benchmark Results")
    print("="*60)
    print(f"{'Optimizer':<15} | {'Test Acc (%)':<12} | {'Conv. Epoch':<11} | {'Best η':<8}")
    print("-" * 55)
    
    for i, (opt_name, res) in enumerate(results.items()):
        if res["curve"] is None:
            continue
            
        # Plot curve
        epochs_range = range(1, EPOCHS + 1)
        ax.plot(epochs_range, res["curve"], marker='o', linewidth=2, 
                color=colors[i], label=f"{opt_name} (η={res['lr']})")
        
        # Table print
        acc_str = f"{res['acc']:.2f}"
        conv_str = f"{res['conv']:.1f}" if res['conv'] is not None else ">EPOCHS"
        lr_str = f"{res['lr']:.2f}"
        
        print(f"{opt_name:<15} | {acc_str:<12} | {conv_str:<11} | {lr_str:<8}")
        
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss (Cross Entropy)")
    ax.set_title("Training Loss vs. Epoch for Ptorch Optimizers")
    ax.legend(loc="best", fontsize=12)
    # ax.set_yscale("log") # optional, uncomment if log scale is preferred
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "optimizer_comparison.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "optimizer_comparison.png"), dpi=300, bbox_inches="tight")
    print(f"\nSaved figures to {out_dir}/optimizer_comparison.pdf/png")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    results = run_grid_search(device)
    plot_and_summarize(results)
