"""Vanishing target signal (paper fig:depth-barrier).

    python -m experiments.neurips.deep.vanishing_target --config experiments/neurips/deep/vanishing_target_config.yaml

Trains L-layer bias-free linear MLPs to recover the identity map with the MSE
projection loss. Then, on one fresh batch through the trained network, computes
the target signal magnitude δ^(i) = ‖h̄^(i) − h^(i)‖₂ for i = 0..L: projected
target minus forward value, the norm taken over the whole stacked batch, with
h^(0) the input. Saves δ and the final MSE to `results_dir`/vanishing_deltas.csv
and plots
  (a) δ per layer, for each depth (vanishing_profile.pdf);
      the same with x counted from the output (vanishing_profile_from_output.pdf);
  (b) final training MSE vs depth (vanishing_final_mse.pdf).
The data is synthetic and fresh every step, so there is no held-out split.
"""
import argparse
import os

import pandas as pd
import torch
import torch.nn as nn

import ptorch.optim_static as poptim
from ptorch.core.ops import MSEProjection
from ptorch.nn.modules import Linear
from experiments.neurips import common, style

style.apply_theme()


class IdentityMLP(nn.Module):
    """Bias-free linear layers."""

    def __init__(self, depth, width):
        super().__init__()
        self.layers = nn.ModuleList(Linear(width, width, bias=False) for _ in range(depth))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def target_signal(model, x):
    """[δ^(0), ..., δ^(L)] of one forward and backward pass on the batch x.
    Changes neither the weights nor their .grad."""
    # h^(0) is its own leaf, apart from the loss target x: MSEProjection also
    # returns a target for its second argument, and on a shared tensor autograd
    # would add that to the input target.
    h = [x.clone().requires_grad_()]
    for layer in model.layers:
        h.append(layer(h[-1]))
    # In projection mode the "gradient" reaching h^(i) is its target h̄^(i).
    targets = torch.autograd.grad(MSEProjection.apply(h[-1], x), h)
    return [(target - value.detach()).norm(p=2).item() for target, value in zip(targets, h)]


def run(cfg, depth, seed, device):
    """Returns ([δ^(0), ..., δ^(L)] of the trained network, final training MSE)."""
    torch.manual_seed(seed)
    model = IdentityMLP(depth, cfg["width"]).to(device)
    opt = cfg["optimizer"]
    optimizer = getattr(poptim, opt["name"])(model.parameters(), **opt.get("kwargs", {}))
    for _ in range(cfg["steps"]):
        x = torch.randn(cfg["batch_size"], cfg["width"], device=device)
        optimizer.zero_grad()
        loss = MSEProjection.apply(model(x), x)
        loss.backward()
        optimizer.step()
    fresh = torch.randn(cfg["batch_size"], cfg["width"], device=device)
    return target_signal(model, fresh), loss.item()


def plot_profile(depths, profiles, from_output):
    """δ^(i), i = 0..L, for each depth; x counts layers from the output or the input."""
    fig, ax = style.subplots()
    for color, depth in zip(style.CYCLE, depths):
        d = torch.tensor(profiles[depth])
        # Stored input-first: flip so that distance 0 is the output layer.
        d = d.flip(1) if from_output else d
        mean, std = d.mean(0), d.std(0)
        # δ is exactly 0 once the target rounds to h in fp32. The log axis would
        # draw that as a vertical drop, so the curve ends there (the CSV keeps it).
        mean = mean.masked_fill(mean == 0, float("nan"))
        x = range(depth + 1)
        ax.plot(x, mean, marker="o", color=color, label=f"$L={depth}$")
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
    ax.set_ylabel(r"$\delta^{(i)}$")
    ax.set_yscale("log")
    ax.set_title("Target Signal Magnitude")
    if from_output:
        ax.set_xlabel("Distance from Output Layer $L - i$")
        ax.set_xticks(range(max(depths) + 1))
        ax.legend(loc="upper right")
    else:
        ax.set_xlabel("Distance from Input Layer")
        ax.legend(loc="lower left")
        ax.invert_xaxis()
    return fig


def save_deltas(cfg, profiles, final_mse):
    """One row per (depth, seed, layer i); `final_mse` repeats within a run."""
    rows = [{"depth": depth, "seed": seed, "layer": i, "delta": delta, "final_mse": mse}
            for depth in cfg["depths"]
            for seed, profile, mse in zip(common.seeds(cfg), profiles[depth], final_mse[depth])
            for i, delta in enumerate(profile)]
    os.makedirs(cfg["results_dir"], exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(cfg["results_dir"], "vanishing_deltas.csv"), index=False)


def plot(cfg, profiles, final_mse):
    depths = cfg["depths"]
    fig1 = plot_profile(depths, profiles, from_output=False)
    fig3 = plot_profile(depths, profiles, from_output=True)
    fig2, ax2 = style.subplots()

    mse = torch.tensor([final_mse[d] for d in depths])
    mean, std = mse.mean(1), mse.std(1)
    ax2.plot(depths, mean, marker="s", color=style.CRIMSON)
    ax2.fill_between(depths, mean - std, mean + std, color=style.CRIMSON, alpha=0.2)
    ax2.set_xlabel("Network Depth $L$")
    ax2.set_ylabel("Final Training Loss (MSE)")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(depths)
    ax2.set_xticklabels(depths)
    ax2.set_title(r"(b) Final Loss vs. Depth $L$")

    os.makedirs(cfg["figures_dir"], exist_ok=True)
    for fig, name in ((fig1, "vanishing_profile.pdf"), (fig3, "vanishing_profile_from_output.pdf"),
                      (fig2, "vanishing_final_mse.pdf")):
        style.savefig(fig, os.path.join(cfg["figures_dir"], name))


def main():
    p = argparse.ArgumentParser(description="Vanishing target signal probe")
    p.add_argument("--config", required=True, help="experiment YAML")
    cfg = common.load_config(p.parse_args().config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    profiles, final_mse = {}, {}
    for depth in cfg["depths"]:
        profiles[depth], final_mse[depth] = [], []
        for seed in common.seeds(cfg):
            profile, mse = run(cfg, depth, seed, device)
            profiles[depth].append(profile)
            final_mse[depth].append(mse)
            print(f"L={depth} seed={seed}: final MSE {mse:.6f}")
    save_deltas(cfg, profiles, final_mse)
    plot(cfg, profiles, final_mse)


if __name__ == "__main__":
    main()
