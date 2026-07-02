"""Stability benchmark: projection policy gradient vs standard (backprop) PG.

Sweeps both methods over a shared geometric learning-rate grid and several
seeds, then reports how each one's final performance and run-to-run volatility
degrade as the learning rate is pushed up. The question is not "which is best at
its tuned lr" but "which tolerates a wider band of learning rates before it
oscillates or collapses" — i.e. stability.

  gradient   : nn.Linear + Tanh, standard Adam, Categorical.log_prob loss
  projection : ptorch.Linear + ReLU, ProjectionAdam, PolicyGradientProjection

Both share the same data-collection loop, batch size, and lr grid. Output:
  - results/pg_stability.csv      (one row per method x lr x seed x epoch)
  - plots/pg_stability.png        (final return + volatility vs lr)
  - a printed summary table.

Runs on LunarLander-v3, a harder benchmark than CartPole: an 8-dim continuous
observation, 4 discrete actions, and a shaped reward that runs negative on
crashes (solving threshold ~200). This widens the dynamic range of returns and
makes the lr-tolerance question more discriminating. Requires the Box2D backend:
`pip install "gymnasium[box2d]"`.
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam

try:
    import gymnasium as gym
except ImportError:
    import gym
Discrete, Box = gym.spaces.Discrete, gym.spaces.Box

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ptorch applies its op overrides on import; we gate them per-run via config.
import ptorch  # noqa: F401
from ptorch.config import config as ptorch_config
from ptorch.nn.modules import Linear as PLinear, ReLU as PReLU
from ptorch.optim_static import ProjectionAdam, ProjectionMuon
from ptorch.core.ops import PolicyGradientProjection

HERE = os.path.dirname(os.path.abspath(__file__))


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_net(method, sizes, norm):
    if method == "projection":
        layers = []
        for j in range(len(sizes) - 1):
            layers.append(PLinear(sizes[j], sizes[j + 1], norm=norm))
            if j < len(sizes) - 2:
                layers.append(PReLU(norm=norm))
        return nn.Sequential(*layers)
    # gradient baseline
    layers = []
    for j in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[j], sizes[j + 1]))
        if j < len(sizes) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def train_run(method, lr, seed, env_name, epochs, batch_size, hidden, fail_floor):
    """Train one policy and return the per-epoch mean-return history."""
    ptorch_config.update("use_projections", method == "projection")
    set_seed(seed)

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n
    net = build_net(method, [obs_dim] + hidden + [n_acts], norm="l2")

    if method == "projection":
        optimizer = ProjectionMuon(net.parameters(), lr=lr)
    else:
        optimizer = Adam(net.parameters(), lr=lr)

    def get_action(obs):
        with torch.no_grad():
            if method == "projection":
                probs = torch.softmax(net(obs), dim=-1)
                return torch.multinomial(probs, 1).item()
            return Categorical(logits=net(obs)).sample().item()

    def compute_loss(obs, act, weights):
        if method == "projection":
            return PolicyGradientProjection.apply(net(obs), act, weights)
        logp = Categorical(logits=net(obs)).log_prob(act)
        return -(logp * weights).mean()

    def train_one_epoch():
        batch_obs, batch_acts, batch_weights, batch_rets = [], [], [], []
        obs, _ = env.reset(seed=seed + len(batch_rets))
        ep_rews = []
        while True:
            batch_obs.append(obs.copy())
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, terminated, truncated, _ = env.step(act)
            batch_acts.append(act)
            ep_rews.append(rew)
            if terminated or truncated:
                ep_ret = sum(ep_rews)
                batch_rets.append(ep_ret)
                batch_weights += [ep_ret] * len(ep_rews)
                obs, _ = env.reset()
                ep_rews = []
                if len(batch_obs) > batch_size:
                    break

        weights = torch.as_tensor(np.array(batch_weights), dtype=torch.float32)
        if method == "projection":
            # standardize returns so the proximal step size stays O(lr)
            weights = (weights - weights.mean()) / (weights.std() + 1e-8)

        optimizer.zero_grad()
        loss = compute_loss(
            torch.as_tensor(np.array(batch_obs), dtype=torch.float32),
            torch.as_tensor(np.array(batch_acts), dtype=torch.int64),
            weights,
        )
        loss.backward()
        optimizer.step()
        return float(np.mean(batch_rets))

    hist = []
    for _ in range(epochs):
        try:
            hist.append(train_one_epoch())
        except Exception:
            # a blown-up run (NaN logits, etc.) is itself a stability signal:
            # record the failure as a floor return for the remaining epochs. On
            # LunarLander 0 is not "bad" (crashes score ~-100..-200), so the
            # floor is a strongly negative crash-level return, not 0.
            hist.extend([fail_floor] * (epochs - len(hist)))
            break
    env.close()
    return hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="LunarLander-v3")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=4000)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    p.add_argument("--fail_floor", type=float, default=-200.0,
                   help="return recorded for the remaining epochs of a diverged "
                        "run (crash-level for LunarLander; use 0 for CartPole)")
    p.add_argument(
        "--lrs", type=float, nargs="+",
        default=[1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0],
    )
    p.add_argument("--methods", nargs="+", default=["gradient", "projection"])
    args = p.parse_args()

    results_dir = os.path.join(HERE, "results")
    plots_dir = os.path.join(HERE, "plots")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    rows = []  # (method, lr, seed, epoch, return)
    for method in args.methods:
        for lr in args.lrs:
            for seed in range(args.seeds):
                hist = train_run(
                    method, lr, seed, args.env,
                    args.epochs, args.batch_size, args.hidden, args.fail_floor,
                )
                for ep, ret in enumerate(hist):
                    rows.append((method, lr, seed, ep, ret))
                tail = np.array(hist[-10:])
                print(f"{method:>10s}  lr={lr:<7g}  seed={seed}  "
                      f"final={tail.mean():7.1f}  volatility={tail.std():6.1f}")

    # ---- write raw CSV ----
    csv_path = os.path.join(results_dir, "pg_stability.csv")
    with open(csv_path, "w") as f:
        f.write("method,lr,seed,epoch,return\n")
        for r in rows:
            f.write("%s,%g,%d,%d,%g\n" % r)

    # ---- aggregate stability metrics ----
    arr = np.array([(m, lr, s, e, r) for (m, lr, s, e, r) in rows], dtype=object)
    half = args.epochs // 2

    def cell(method, lr, seed):
        return [r for (m, l, s, e, r) in rows
                if m == method and l == lr and s == seed]

    summary = {}  # method -> dict of arrays over lr
    print("\n%-10s %-8s %12s %12s %12s" %
          ("method", "lr", "final_mean", "final_std", "volatility"))
    print("-" * 58)
    for method in args.methods:
        finals_mean, finals_std, vols = [], [], []
        for lr in args.lrs:
            per_seed_final, per_seed_vol = [], []
            for seed in range(args.seeds):
                h = np.array(cell(method, lr, seed))
                per_seed_final.append(h[-10:].mean())
                per_seed_vol.append(h[half:].std())  # within-run oscillation
            fm = float(np.mean(per_seed_final))
            fs = float(np.std(per_seed_final))      # across-seed spread
            vv = float(np.mean(per_seed_vol))
            finals_mean.append(fm)
            finals_std.append(fs)
            vols.append(vv)
            print("%-10s %-8g %12.1f %12.1f %12.1f" % (method, lr, fm, fs, vv))
        summary[method] = dict(
            final=np.array(finals_mean),
            fstd=np.array(finals_std),
            vol=np.array(vols),
        )

    # ---- plot ----
    lrs = np.array(args.lrs)
    colors = {"gradient": "tab:red", "projection": "tab:blue"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for method in args.methods:
        c = colors.get(method, None)
        s = summary[method]
        axes[0].plot(lrs, s["final"], "-o", color=c, label=method)
        axes[0].fill_between(lrs, s["final"] - s["fstd"],
                             s["final"] + s["fstd"], color=c, alpha=0.2)
        axes[1].plot(lrs, s["vol"], "-o", color=c, label=method)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("learning rate")
    axes[0].set_ylabel("final return (mean of last 10 epochs)")
    axes[0].set_title("Performance vs learning rate (band = ±std across seeds)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].set_xscale("log")
    axes[1].set_xlabel("learning rate")
    axes[1].set_ylabel("within-run volatility (std of returns, 2nd half)")
    axes[1].set_title("Instability vs learning rate (lower = more stable)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.suptitle("Policy-gradient stability: projection vs backprop (%s)" % args.env)
    fig.tight_layout()
    out = os.path.join(plots_dir, "pg_stability.png")
    fig.savefig(out, dpi=120)
    print("\nSaved CSV  -> %s" % csv_path)
    print("Saved plot -> %s" % out)


if __name__ == "__main__":
    main()
