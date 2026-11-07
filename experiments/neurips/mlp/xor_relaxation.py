"""First-order relaxation on the XOR benchmark (paper fig:relaxation-location).

    python -m experiments.neurips.mlp.xor_relaxation --config experiments/neurips/mlp/xor_relaxation_config.yaml

Cyclic projections written out by hand on a 2-8-1 network, so that the relaxation can
also act on the hidden states, which the Torch-native framework does not expose. Four
variants: plain cyclic projections, relaxation of the weights, of the weights and hidden
states, and of the weights with the hidden states reset by a forward pass every iteration.
Saves results_dir/xor_error.csv and xor_activations.pdf.
"""
import argparse
import os

import pandas as pd
import seaborn as sns
import torch

from experiments.neurips import common, style

style.apply_theme()

DELTA = 1.0   # output margin of the positive class
SLOPE = 0.1   # LeakyReLU slope
METHODS = {"cyclic": "Cyclic Projections",
           "weights": "Relaxation (Weights)",
           "activations": "Relaxation (Weights + Activations)",
           "forward": "Relaxation (Weights + Forward Passes)"}


def xor_data():
    """X (2, 4): every pair of bits; y (1, 4): their parity."""
    idx = torch.arange(4)
    X = torch.stack([idx & 1, (idx >> 1) & 1])
    return X.float(), (X.sum(0, keepdim=True) % 2).float()


def forward(W0, W1, X, y):
    """Returns (mean squared margin violation, pre-activation, activation, output)."""
    pre = W0 @ X
    act = torch.where(pre > 0, pre, SLOPE * pre)
    out = W1 @ act
    err = torch.where(y == 1, (out - DELTA).clamp(max=0) ** 2, out.clamp(min=0) ** 2)
    return err.mean().item(), pre, act, out


def project_output(out, y):
    return torch.where(y == 1, out.clamp(min=DELTA), out.clamp(max=0.0))


def project_matmul(X, W, Z, num_steps=10):
    """Fused projection onto W @ X = Z: returns the consensus targets of X and W, and Z."""
    p = W @ X
    q = (W * W).sum(1, keepdim=True) + (X * X).sum(0, keepdim=True)
    eps, damping, max_step = 1e-5, 1e-4, 0.5
    t_max = (1.0 - eps) ** 0.5
    t = torch.zeros_like(p)
    for _ in range(num_steps):
        t = t.clamp(-t_max, t_max)
        t2 = t * t
        d = 1.0 - t2
        num = p * (1.0 + t2) + t * q
        f = num / d ** 2 - Z + t
        fp = ((2.0 * t * p + q) * d + 4.0 * t * num) / d ** 3 + 1.0
        t = t - (f / (fp.abs() + damping)).clamp(-max_step, max_step)
    t = t.clamp(-t_max, t_max)
    u, v = 1.0 / (1.0 - t * t), t / (1.0 - t * t)
    W_bar = (W * u.sum(1, keepdim=True) + v @ X.T) / X.shape[1]
    X_bar = (X * u.sum(0, keepdim=True) + W.T @ v) / W.shape[0]
    return X_bar, W_bar, Z - t


def project_input_matmul(X, W, Z):
    """Projection onto W @ X = Z with X fixed (the data): returns the targets of W and Z."""
    S = X.T @ X
    Z_bar = torch.linalg.solve(torch.eye(S.shape[0]) + S, (W @ X + Z @ S).T).T
    return W - (Z_bar - Z) @ X.T, Z_bar


def project_leaky_relu(x, z):
    """Closest point of the LeakyReLU graph to (x, z): the closer of its two rays."""
    x1 = ((x + SLOPE * z) / (1.0 + SLOPE ** 2)).clamp(max=0.0)
    y1 = SLOPE * x1
    x2 = ((x + z) / 2.0).clamp(min=0.0)
    inactive = (x - x1) ** 2 + (z - y1) ** 2 < (x - x2) ** 2 + (z - x2) ** 2
    return torch.where(inactive, x1, x2), torch.where(inactive, y1, x2)


class Relax:
    """One SGD-with-momentum step on the residual x - target of a tensor."""

    def __init__(self, lr, momentum):
        self.lr, self.momentum, self.buf = lr, momentum, 0.0

    def __call__(self, x, target):
        self.buf = (x - target) + self.momentum * self.buf
        return x - self.lr * self.buf


def run(cfg, method, seed, iterations):
    """Returns the error before training and after each iteration."""
    X, y = xor_data()
    gen = torch.Generator().manual_seed(seed)
    W0 = torch.randn(cfg["hidden"], X.shape[0], generator=gen)
    W1 = torch.randn(1, cfg["hidden"], generator=gen)
    relax_W0, relax_W1, relax_pre, relax_act = (Relax(**cfg["optimizer"]) for _ in range(4))
    plain = lambda x, target: target
    step_W0, step_W1 = (plain, plain) if method == "cyclic" else (relax_W0, relax_W1)
    step_pre, step_act = (relax_pre, relax_act) if method == "activations" else (plain, plain)

    err, pre, act, out = forward(W0, W1, X, y)
    errors = [err]
    for _ in range(iterations):
        out = project_output(out, y)
        act_bar, W1_bar, out = project_matmul(act, W1, out)
        W1 = step_W1(W1, W1_bar)
        act = step_act(act, act_bar)
        pre_bar, act_bar = project_leaky_relu(pre, act)
        act = step_act(act, act_bar)
        pre = step_pre(pre, pre_bar)
        W0_bar, pre_bar = project_input_matmul(X, W0, pre)
        W0 = step_W0(W0, W0_bar)
        pre = step_pre(pre, pre_bar)
        err, *state = forward(W0, W1, X, y)
        errors.append(err)
        if method == "forward":
            pre, act, out = state
    return errors


def plot(cfg, df):
    fig, ax = style.subplots()
    sns.lineplot(data=df, x="iteration", y="error", hue="method", hue_order=list(METHODS.values()),
                 palette=style.sequential(len(METHODS)), estimator="mean", errorbar=("sd", 1), ax=ax)
    ax.set_yscale("log")
    ax.set_title("XOR Classification Convergence Analysis")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Error")
    ax.legend(loc="lower left")
    os.makedirs(cfg["figures_dir"], exist_ok=True)
    style.savefig(fig, os.path.join(cfg["figures_dir"], "xor_activations.pdf"))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, help="experiment YAML")
    p.add_argument("--max-steps", type=int, help="smoke test: iterations per run")
    args = p.parse_args()
    cfg = common.load_config(args.config)
    iterations = args.max_steps or cfg["iterations"]

    rows = [{"method": label, "seed": seed, "iteration": i, "error": err}
            for method, label in METHODS.items() for seed in common.seeds(cfg)
            for i, err in enumerate(run(cfg, method, seed, iterations))]
    df = pd.DataFrame(rows)
    os.makedirs(cfg["results_dir"], exist_ok=True)
    df.to_csv(os.path.join(cfg["results_dir"], "xor_error.csv"), index=False)
    plot(cfg, df)


if __name__ == "__main__":
    main()
