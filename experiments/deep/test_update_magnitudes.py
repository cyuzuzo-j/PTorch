################################################
###   Update magnitude visualisation        ###
###   (per-parameter, per-step)             ###
################################################
"""
Runs a short training loop for each optimizer and records:
  1. ||p_new - p_old||  (weight update norm) at every step.
  2. ||layer_out_new - layer_out_old||  (layer projection / output change norm)
     for every LinearBias layer at every step.

Two figures are saved:
  update_magnitudes.png       – weight update norms
  output_change_magnitudes.png – layer output change norms
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))

import torch
import torch.nn as tnn
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless – works on cluster nodes
import matplotlib.pyplot as plt

from ptorch.nn.modules import LinearBias, ReLU
from ptorch.core.ops import MarginLossProjection
from ptorch.optim_static import (
    AlternatingProjections,
    ProjectionSGD,
    ProjectionAdam,
    ProjectionAdagrad,
    ProjectionAdadelta,
)
from experiments.shared.data import MNISTDataModule

# ─── Configuration ────────────────────────────────────────────────────────────
RANDOM_SEED  = 42
BATCH_SIZE   = 256
MAX_STEPS    = 100          # keep short – we just want to see update dynamics
HIDDEN       = [10,10,10,10]

# Alpha schedule: alpha_top at output layer, exponentially decayed toward 0 at
# the input layer.  alpha(layer_i) = alpha_top * exp(-decay_rate * (n-1 - i))
# where layer_i=0 is the input-side layer and layer_i=n-1 is the output layer.
ALPHA_TOP   = 1e9    # alpha for the output (top) layer
DECAY_RATE  = 1   # controls how fast alpha falls off toward the input

# Each entry: (display_name, optimizer_class_or_None, optimizer_kwargs, alpha_override)
# alpha_override=None       → use the exponential decay schedule (ALPHA_TOP / DECAY_RATE)
# alpha_override=float      → all layers get the same fixed alpha (no decay)
# alpha_override="all_but_input" → all layers get ALPHA_TOP except input layer (α=1)
OPTIMIZERS = [
    ("AP α=1 (weights free)",      AlternatingProjections, {}, 1.0),
    ("AP α=1e6 (activations stiff)", AlternatingProjections, {}, 1e6),
    ("AP all-but-input α=1e9",     AlternatingProjections, {}, "all_but_input"),
]

OUT_FILE         = os.path.join(os.path.dirname(__file__), "update_magnitudes.png")
OUT_FILE_OUTPUTS = os.path.join(os.path.dirname(__file__), "output_change_magnitudes.png")


def make_alphas(n_layers: int, alpha_top: float = ALPHA_TOP,
                decay_rate: float = DECAY_RATE) -> list[float]:
    """
    Returns a list of n_layers alpha values.
    Index 0 = input-side layer  → smallest alpha (≈ 0)
    Index n-1 = output layer    → alpha_top

    alpha[i] = alpha_top * exp(-decay_rate * (n_layers - 1 - i))
    """
    return [alpha_top * float(np.exp(-decay_rate * (n_layers - 1 - i)))
            for i in range(n_layers)]


def make_alphas_all_but_input(n_layers: int, alpha_top: float = ALPHA_TOP,
                              input_alpha: float = 1.0) -> list[float]:
    """
    Returns a list of n_layers alpha values where every layer except the
    input-side layer (index 0) gets alpha_top, and the input layer gets
    input_alpha (a small value).
    """
    alphas = [alpha_top] * n_layers
    alphas[0] = input_alpha
    return alphas


# ─── Model ────────────────────────────────────────────────────────────────────
class MLP(tnn.Module):
    def __init__(self, hidden_features, in_features, classes,
                 alpha_top: float = ALPHA_TOP, decay_rate: float = DECAY_RATE):
        super().__init__()
        self.hidden_features = hidden_features

        # Total linear layers = len(hidden_features) hidden + 1 output
        n_linear = len(hidden_features) + 1
        alphas = make_alphas(n_linear, alpha_top=alpha_top, decay_rate=decay_rate)

        last_f = in_features
        self.layers = tnn.ModuleList()
        for layer_idx, f in enumerate(hidden_features):
            self.layers.append(LinearBias(last_f, f, alpha=alphas[layer_idx]))
            self.layers.append(ReLU(f))
            last_f = f

        # Output layer gets the highest alpha (alphas[-1])
        self.out = LinearBias(hidden_features[-1], classes, alpha=alphas[-1])

        # Expose the schedule for inspection / logging
        self.alphas = alphas

    def forward(self, x, record_outputs: bool = False):
        x = x.reshape(x.shape[0], -1)
        self._layer_outputs: list = []   # populated when record_outputs=True
        for i in range(0, len(self.layers), 2):
            x = self.layers[i](x)
            if record_outputs:
                self._layer_outputs.append(x.detach().clone())
            x = self.layers[i + 1](x)
        x = self.out(x)
        if record_outputs:
            self._layer_outputs.append(x.detach().clone())
        return x


# ─── One training run, returns dict: param_name → [update_norm per step] ─────
def collect_update_norms(opt_name, opt_class, opt_kwargs, device, alpha_override=None):
    torch.manual_seed(RANDOM_SEED)

    dataset   = MNISTDataModule(batch_size=BATCH_SIZE, seed=RANDOM_SEED)
    train_iter = dataset.train_iterator()

    if alpha_override == "all_but_input":
        # All layers get ALPHA_TOP except the input-side layer (α=1)
        n_linear = len(HIDDEN) + 1
        custom_alphas = make_alphas_all_but_input(n_linear, alpha_top=ALPHA_TOP, input_alpha=1.0)
        model = MLP(HIDDEN, 28 * 28, 10,
                    alpha_top=ALPHA_TOP, decay_rate=0.0).to(device)
        # Override the alphas directly on each LinearBias layer
        linear_idx = 0
        for layer in list(model.layers) + [model.out]:
            if hasattr(layer, 'alpha'):
                layer.alpha = custom_alphas[linear_idx]
                linear_idx += 1
        model.alphas = custom_alphas
    elif alpha_override is not None:
        # Flat schedule: every layer gets the same alpha
        model = MLP(HIDDEN, 28 * 28, 10,
                    alpha_top=alpha_override, decay_rate=0.0).to(device)
    else:
        model = MLP(HIDDEN, 28 * 28, 10).to(device)

    # None opt_class → no-update baseline (parameters never change)
    optimizer = opt_class(model.parameters(), **opt_kwargs) if opt_class is not None else None

    # Print the alpha schedule for this run
    alphas = model.alphas
    print(f"  Alpha schedule (input→output): "
          + "  ".join(f"L{i}={a:.3g}" for i, a in enumerate(alphas)))

    # Map param id → human-readable name
    param_names = {id(p): name for name, p in model.named_parameters()}

    # LinearBias layers in forward order: hidden[0..n-2], then output
    linear_layers = [model.layers[i] for i in range(0, len(model.layers), 2)] + [model.out]
    n_linear      = len(linear_layers)   # == len(HIDDEN) + 1
    # aproj_names[i] = projected activation change AT the OUTPUT of linear_layers[i]
    # (i.e. A_proj returned by linear_layers[i+1].backward, flowing into layer i's output)
    # For the last layer the incoming A_proj comes from MarginLossProjection.
    aproj_names = [f"L{i}_aproj" for i in range(n_linear)]

    history       = {name: [] for name in param_names.values()}
    aproj_history = {name: [] for name in aproj_names}

    model.train()
    for step in range(MAX_STEPS):
        x, y = next(train_iter)
        x = torch.tensor(x, dtype=torch.float32, device=device)
        y = torch.tensor(y, dtype=torch.long,    device=device)

        # Snapshot parameter values BEFORE the step
        before = {id(p): p.data.clone() for p in model.parameters()}

        # ── forward + backward with hooks to capture A_proj per layer ────────
        # Strategy: one forward pass with a module hook that captures both
        # A_old (detached value) and the live differentiable output tensor.
        # A grad hook on the live tensor then intercepts A_proj during backward.
        captured_aproj = {}   # layer_index → ||A_proj - A_old||
        hook_handles   = []

        def make_fwd_hook(idx):
            def hook(module, inp, out):
                a_old = out.detach().clone()
                def make_grad_hook(a_old=a_old):
                    def grad_hook(grad):   # grad == A_proj flowing into this output
                        captured_aproj[idx] = (grad - a_old).norm().item()
                    return grad_hook
                if out.requires_grad:
                    hook_handles.append(out.register_hook(make_grad_hook()))
            return hook

        fwd_handles = [
            layer.register_forward_hook(make_fwd_hook(i))
            for i, layer in enumerate(linear_layers)
        ]

        logits = model(x)
        for h in fwd_handles:
            h.remove()

        y_one_hot = tnn.functional.one_hot(y, num_classes=logits.shape[-1]).float()
        projected = MarginLossProjection.apply(logits, y_one_hot)

        optimizer.zero_grad() if optimizer is not None else model.zero_grad()
        projected.sum().backward()
        if optimizer is not None:
            optimizer.step()

        for h in hook_handles:
            h.remove()
        # ────────────────────────────────────────────────────────────────────

        # Record ||p_new - p_old||_2  for every parameter
        with torch.no_grad():
            for p in model.parameters():
                norm = (p.data - before[id(p)]).norm().item()
                history[param_names[id(p)]].append(norm)

        # Record ||A_proj - A_old||_2 per layer (captured by hooks above)
        for i, aname in enumerate(aproj_names):
            aproj_history[aname].append(captured_aproj.get(i, 0.0))

    return history, aproj_history


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("Starting main", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Collect results
    all_results       = {}
    all_aproj_results = {}
    for opt_name, opt_class, opt_kwargs, alpha_override in OPTIMIZERS:
        print(f"Running {opt_name} …")
        history, aproj_history = collect_update_norms(
            opt_name, opt_class, opt_kwargs, device, alpha_override=alpha_override
        )
        all_results[opt_name]       = history
        all_aproj_results[opt_name] = aproj_history

    # ── Build parameter name list (same for every optimizer) ──────────────
    param_names = list(next(iter(all_results.values())).keys())
    n_params    = len(param_names)
    n_opts      = len(OPTIMIZERS)
    steps       = list(range(MAX_STEPS))

    # ── Plot ──────────────────────────────────────────────────────────────
    # One row per parameter, one column per optimizer.
    fig, axes = plt.subplots(
        n_params, n_opts,
        figsize=(4 * n_opts, 2.5 * n_params),
        sharex=True,
        squeeze=False,
    )

    # Compute per-row (per-param) y-max across all optimizers for fair comparison
    row_ymaxes = []
    for pi, pname in enumerate(param_names):
        ymax = max(
            max(all_results[opt_name][pname], default=0)
            for opt_name, _, _, _ in OPTIMIZERS
        )
        row_ymaxes.append(ymax * 1.05 + 1e-9)   # tiny floor so empty axes work

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for oi, (opt_name, _, _, _) in enumerate(OPTIMIZERS):
        for pi, pname in enumerate(param_names):
            ax    = axes[pi][oi]
            norms = all_results[opt_name][pname]

            ax.plot(steps, norms, linewidth=0.8, color=colors[oi], alpha=0.85)
            ax.fill_between(steps, norms, alpha=0.15, color=colors[oi])

            ax.set_ylim(0, row_ymaxes[pi])
            ax.grid(True, linewidth=0.4, alpha=0.5)

            # Column header (optimizer name) on top row
            if pi == 0:
                ax.set_title(opt_name, fontsize=9, fontweight="bold", pad=4)

            # Row label (parameter name) on left column
            if oi == 0:
                # Shorten long param names for readability
                short = pname.replace("layers.", "L").replace(".weight", ".W")
                ax.set_ylabel(short, fontsize=7, labelpad=3, rotation=0,
                              ha="right", va="center")

            # X-axis label on bottom row
            if pi == n_params - 1:
                ax.set_xlabel("step", fontsize=8)

    # Build alpha schedule string for title (from a temporary model)
    _tmp = MLP(HIDDEN, 28 * 28, 10)
    alpha_str = ", ".join(f"{a:.2g}" for a in _tmp.alphas)

    fig.suptitle(
        f"Parameter update magnitude  ||Δp||₂  per step\n"
        f"(MLP {HIDDEN}, MNIST, batch={BATCH_SIZE}, steps={MAX_STEPS})\n"
        f"α (input→output): [{alpha_str}]  (top={ALPHA_TOP:.0e}, decay={DECAY_RATE})",
        fontsize=10, y=1.005,
    )
    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {OUT_FILE}")

    # ── Figure 2: Projected activation change norms (A_proj vs A_old) ───────
    n_linear     = len(HIDDEN) + 1
    aproj_names  = [f"L{i}_aproj" for i in range(n_linear)]
    n_layers_fig = len(aproj_names)

    fig2, axes2 = plt.subplots(
        n_layers_fig, n_opts,
        figsize=(4 * n_opts, 2.5 * n_layers_fig),
        sharex=True,
        squeeze=False,
    )

    row_ymaxes2 = []
    for li, aname in enumerate(aproj_names):
        ymax = max(
            max(all_aproj_results[opt_name][aname], default=0)
            for opt_name, _, _, _ in OPTIMIZERS
        )
        row_ymaxes2.append(ymax * 1.05 + 1e-9)

    for oi, (opt_name, _, _, _) in enumerate(OPTIMIZERS):
        for li, aname in enumerate(aproj_names):
            ax    = axes2[li][oi]
            norms = all_aproj_results[opt_name][aname]

            ax.plot(steps, norms, linewidth=0.8, color=colors[oi], alpha=0.85)
            ax.fill_between(steps, norms, alpha=0.15, color=colors[oi])

            ax.set_ylim(0, row_ymaxes2[li])
            ax.grid(True, linewidth=0.4, alpha=0.5)

            if li == 0:
                ax.set_title(opt_name, fontsize=9, fontweight="bold", pad=4)

            if oi == 0:
                label = f"L{li} Aproj" if li < n_layers_fig - 1 else "out Aproj"
                ax.set_ylabel(label, fontsize=7, labelpad=3, rotation=0,
                              ha="right", va="center")

            if li == n_layers_fig - 1:
                ax.set_xlabel("step", fontsize=8)

    fig2.suptitle(
        f"Projected activation change  ||A_proj − A_old||₂  per step\n"
        f"(MLP {HIDDEN}, MNIST, batch={BATCH_SIZE}, steps={MAX_STEPS})\n"
        f"α (input→output): [{alpha_str}]  (top={ALPHA_TOP:.0e}, decay={DECAY_RATE})",
        fontsize=10, y=1.005,
    )
    plt.tight_layout()
    plt.savefig(OUT_FILE_OUTPUTS, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_FILE_OUTPUTS}")

    # Also print a quick summary table
    print(f"\n{'Optimizer':<30} {'Param':<30} {'mean Δ':>10} {'max Δ':>10}")
    print("-" * 82)
    for opt_name, _, _, _ in OPTIMIZERS:
        for pname in param_names:
            norms = all_results[opt_name][pname]
            print(f"{opt_name:<30} {pname:<30} {np.mean(norms):>10.4f} {np.max(norms):>10.4f}")

    print(f"\n{'Optimizer':<30} {'Layer (A_proj)':<20} {'mean Δ':>10} {'max Δ':>10}")
    print("-" * 62)
    for opt_name, _, _, _ in OPTIMIZERS:
        for aname in aproj_names:
            norms = all_aproj_results[opt_name][aname]
            print(f"{opt_name:<30} {aname:<20} {np.mean(norms):>10.4f} {np.max(norms):>10.4f}")


if __name__ == "__main__":
    main()
