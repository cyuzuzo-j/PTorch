"""
TD(λ) Eligibility Trace vs. Vanishing Targets
=============================================
Identity-recovery task (Section 5.2 testbed): linear MLPs of depth L ∈ {2,4,8},
width 32, trained on f(x) = x with MSE projection loss, sweeping the
depth-trace parameter td_lambda (λ_TD) ∈ {0, 0.3, 0.5, 0.7, 0.9, 1}.

Two trace variants (--mode, config.td_mode):
  vector — Variant A: blend the residual *vector* down the sweep (frame-
           misaligned identity transport; measured cos ~ 1/sqrt(d) with the
           useful direction; negative control).
  norm   — Variant B: carry only a scalar magnitude floor down the sweep and
           rescale the local (Bᵀ-transported, well-aligned) direction up to it.

Hardened testbed (--freeze-top K): freeze the top K layers at random init so
the bottom layers *must* learn — otherwise the top layer solves linear identity
alone and final MSE cannot reflect propagation quality.

Logged per layer at logged steps:
  g_hook      — ‖delivered target − activation‖/B at each layer output
  delta_local — ‖r_local‖ of the pure local projection residual (δ^(i))
  g_trace     — ‖signal actually written‖ after the trace
  cos         — alignment of the delivered residual with the true backprop
                gradient at that hidden state (the direction-quality metric)
plus the training loss per logged step.

Plots: (a) step-1 signal profile, (b) final-MSE-vs-depth, (c) step-1 alignment
profile, (d) loss curves at the largest depth.
"""

import argparse
import csv
import os
import random
import sys

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vanishing_target import IdentityMLP, D, BATCH_SIZE

from ptorch.config import config
from ptorch.core import ops
from ptorch.optim_static import ProjectionMuonV2
from ptorch.core.ops import MSEProjection

sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")


class TDTraceSink:
    """ops.DIAG sink collecting the per-backward-call td_trace records.

    Within one backward sweep the calls arrive deep→shallow: call k belongs to
    the Linear whose input is hidden state h_{L-1-k} (h_0 = network input).
    """

    def __init__(self):
        self.records = []

    def record(self, key, **scalars):
        if key == "td_trace":
            self.records.append(scalars)

    def drain(self):
        out, self.records = self.records, []
        return out


def measure_alignment(model, x, y, sink):
    """cos(delivered residual, true backprop gradient) per hidden state 1..L.

    Runs one projection-mode and one gradient-mode forward/backward on the
    same weights; pollutes captured_deltas / param grads / TD_TRACE, so call
    it BEFORE the real training pass of the step and zero grads after.
    """
    def collect(proj):
        config.update("use_projections", proj)
        hs = []
        h = x
        for layer in model.layers:
            h = layer(h)
            h.retain_grad()
            hs.append(h)
        loss = MSEProjection.apply(h, y) if proj else F.mse_loss(h, y)
        loss.backward()
        return hs

    # The measurement's solves would overwrite each layer's warm-started
    # proj_cache ('t'); a stale/saturated t poisons the next 1-step Newton
    # solve, so snapshot and restore around the measurement.
    saved_caches = [dict(layer.proj_cache) for layer in model.layers]

    hs_proj = collect(True)
    residuals = [(h.detach() - h.grad.detach()) for h in hs_proj]
    model.zero_grad(set_to_none=True)
    hs_grad = collect(False)
    grads = [h.grad.detach().clone() for h in hs_grad]
    model.zero_grad(set_to_none=True)
    config.update("use_projections", True)
    sink.drain()  # discard td_trace records from the measurement pass
    for layer, cache in zip(model.layers, saved_caches):
        layer.proj_cache.clear()
        layer.proj_cache.update(cache)

    coss = []
    for r, g in zip(residuals, grads):
        denom = float(r.norm() * g.norm())
        coss.append(float(torch.dot(r.flatten(), g.flatten())) / denom
                    if denom > 1e-30 else 0.0)
    return coss


def run_experiment(lams, depths, n_seeds, n_steps, log_every, td_clip, td_mode,
                   freeze_top, device):
    trace_rows, hook_rows, final_rows, align_rows, loss_rows = [], [], [], [], []
    sink = TDTraceSink()
    ops.DIAG = sink

    config.update("td_mode", td_mode)
    config.update("td_clip", float(td_clip))
    for lam in lams:
        config.update("td_lambda", float(lam))
        for depth in depths:
            for seed in range(n_seeds):
                torch.manual_seed(seed)
                random.seed(seed)

                model = IdentityMLP(depth=depth, d=D).to(device)
                if freeze_top > 0:
                    for layer in model.layers[depth - freeze_top:]:
                        for p in layer.parameters():
                            p.requires_grad_(False)
                trainable = [p for p in model.parameters() if p.requires_grad]
                optimizer = ProjectionMuonV2(trainable)
                final_loss = 0.0

                for step in range(1, n_steps + 1):
                    x_batch = torch.randn(BATCH_SIZE, D, device=device)
                    y_batch = x_batch.clone()

                    logged = step == 1 or step == n_steps or step % log_every == 0

                    if logged:
                        coss = measure_alignment(model, x_batch, y_batch, sink)
                        for i, c in enumerate(coss, start=1):
                            align_rows.append({
                                "lam": lam, "depth": depth, "seed": seed,
                                "step": step, "hidden_idx": i, "cos": c,
                            })

                    model.train()
                    optimizer.zero_grad()

                    preds = model(x_batch)
                    loss = MSEProjection.apply(preds, y_batch)
                    loss.backward()

                    records = sink.drain()
                    if logged:
                        for k, rec in enumerate(records):
                            trace_rows.append({
                                "lam": lam, "depth": depth, "seed": seed,
                                "step": step, "hidden_idx": depth - 1 - k,
                                "delta_local": rec["r_local_norm"],
                                "g_trace": rec["g_norm"],
                            })
                        for j in range(depth):
                            hook_rows.append({
                                "lam": lam, "depth": depth, "seed": seed,
                                "step": step, "layer": j,
                                "g_hook": model.captured_deltas.get(j, 0.0),
                            })
                        loss_rows.append({
                            "lam": lam, "depth": depth, "seed": seed,
                            "step": step, "loss": loss.item() / BATCH_SIZE,
                        })

                    optimizer.step()
                    if step == n_steps:
                        final_loss = loss.item() / BATCH_SIZE

                final_rows.append({
                    "lam": lam, "depth": depth, "seed": seed,
                    "final_mse": final_loss,
                })
                print(f"mode={td_mode} lam={lam:<4} L={depth} seed={seed} | "
                      f"final MSE: {final_loss:.6e}")

    ops.DIAG = None
    config.update("td_lambda", 0.0)
    config.update("td_clip", 0.0)
    config.update("td_mode", "vector")
    return trace_rows, hook_rows, final_rows, align_rows, loss_rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def _mean_over_seeds(rows, keys, value):
    """Group rows by `keys` tuple and average `value`."""
    acc = {}
    for r in rows:
        k = tuple(r[key] for key in keys)
        acc.setdefault(k, []).append(r[value])
    return {k: sum(v) / len(v) for k, v in acc.items()}


def plot_results(hook_rows, final_rows, align_rows, loss_rows, lams, depths,
                 n_steps, tag):
    os.makedirs(IMAGES_DIR, exist_ok=True)
    colors = sns.color_palette("viridis", len(lams))

    # (a) step-1 profile of the delivered signal, one panel per depth
    step1 = _mean_over_seeds(
        [r for r in hook_rows if r["step"] == 1],
        ["lam", "depth", "layer"], "g_hook",
    )
    fig, axes = plt.subplots(1, len(depths), figsize=(5 * len(depths), 4.5),
                             sharey=True)
    axes = [axes] if len(depths) == 1 else list(axes)
    for ax, depth in zip(axes, depths):
        for c, lam in zip(colors, lams):
            ys = [step1.get((lam, depth, j), float("nan")) for j in range(depth)]
            ax.plot(range(1, depth + 1), ys, marker="o", color=c,
                    label=rf"$\lambda_{{TD}}={lam}$")
        ax.set_xlabel("Distance from Input Layer")
        ax.set_yscale("log")
        ax.set_title(rf"$L={depth}$")
        ax.invert_xaxis()
    axes[0].set_ylabel(r"Delivered Signal $\|g\|$ (Step 1)")
    axes[-1].legend(fontsize=9, loc="best")
    fig.suptitle(rf"(a) Step-1 Signal Profile ({tag})")
    fig.tight_layout()
    path_a = os.path.join(IMAGES_DIR, f"td_{tag}_step1_profile.pdf")
    fig.savefig(path_a, dpi=300, bbox_inches="tight")

    # (b) final MSE vs depth, one curve per lambda
    finals = _mean_over_seeds(final_rows, ["lam", "depth"], "final_mse")
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))
    for c, lam in zip(colors, lams):
        ys = [finals.get((lam, d), float("nan")) for d in depths]
        ax2.plot(depths, ys, marker="s", color=c,
                 label=rf"$\lambda_{{TD}}={lam}$")
    ax2.set_xlabel("Network Depth $L$")
    ax2.set_ylabel(f"Final Training Loss (MSE, {n_steps} steps)")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xticks(depths)
    ax2.set_xticklabels(depths)
    ax2.set_title(rf"(b) Final Loss vs. Depth ({tag})")
    ax2.legend(fontsize=9)
    fig2.tight_layout()
    path_b = os.path.join(IMAGES_DIR, f"td_{tag}_final_mse.pdf")
    fig2.savefig(path_b, dpi=300, bbox_inches="tight")

    # (c) step-1 alignment profile
    al1 = _mean_over_seeds(
        [r for r in align_rows if r["step"] == 1],
        ["lam", "depth", "hidden_idx"], "cos",
    )
    fig3, axes3 = plt.subplots(1, len(depths), figsize=(5 * len(depths), 4.5),
                               sharey=True)
    axes3 = [axes3] if len(depths) == 1 else list(axes3)
    for ax, depth in zip(axes3, depths):
        for c, lam in zip(colors, lams):
            ys = [al1.get((lam, depth, i), float("nan"))
                  for i in range(1, depth + 1)]
            ax.plot(range(1, depth + 1), ys, marker="o", color=c,
                    label=rf"$\lambda_{{TD}}={lam}$")
        ax.set_xlabel("Hidden State Index")
        ax.set_ylim(-0.3, 1.05)
        ax.axhline(0.0, color="gray", lw=0.8)
        ax.set_title(rf"$L={depth}$")
    axes3[0].set_ylabel("cos(delivered, true gradient) — Step 1")
    axes3[-1].legend(fontsize=9, loc="best")
    fig3.suptitle(rf"(c) Direction Quality of the Delivered Signal ({tag})")
    fig3.tight_layout()
    path_c = os.path.join(IMAGES_DIR, f"td_{tag}_alignment.pdf")
    fig3.savefig(path_c, dpi=300, bbox_inches="tight")

    # (d) loss curves at the largest depth
    dmax = max(depths)
    curves = _mean_over_seeds(
        [r for r in loss_rows if r["depth"] == dmax],
        ["lam", "step"], "loss",
    )
    fig4, ax4 = plt.subplots(figsize=(6.5, 4.5))
    steps_sorted = sorted({k[1] for k in curves})
    for c, lam in zip(colors, lams):
        ys = [curves.get((lam, s), float("nan")) for s in steps_sorted]
        ax4.plot(steps_sorted, ys, color=c, label=rf"$\lambda_{{TD}}={lam}$")
    ax4.set_xlabel("Training Step")
    ax4.set_ylabel("Training Loss (MSE)")
    ax4.set_yscale("log")
    ax4.set_title(rf"(d) Convergence at $L={dmax}$ ({tag})")
    ax4.legend(fontsize=9)
    fig4.tight_layout()
    path_d = os.path.join(IMAGES_DIR, f"td_{tag}_loss_curves.pdf")
    fig4.savefig(path_d, dpi=300, bbox_inches="tight")

    print(f"Saved plots: {path_a}, {path_b}, {path_c}, {path_d}")

    # Instability check: does the delivered step-1 signal grow toward the input?
    for depth in depths:
        for lam in lams:
            top = step1.get((lam, depth, depth - 1))
            bottom = step1.get((lam, depth, 0))
            if top and bottom and bottom > 1.5 * top:
                print(f"WARNING: ||g|| grows toward the input at "
                      f"lam={lam}, L={depth} (bottom/top = {bottom / top:.2f}) "
                      f"— feasibility-guarantee breakdown regime.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--lams", type=float, nargs="+",
                        default=[0.0, 0.3, 0.5, 0.7, 0.9, 1.0])
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--mode", type=str, default="vector",
                        choices=["vector", "norm"],
                        help="trace variant (config.td_mode)")
    parser.add_argument("--freeze-top", type=int, default=0,
                        help="freeze the top K layers at random init so the "
                             "bottom layers must learn")
    parser.add_argument("--clip", type=float, default=0.0,
                        help="td_clip safeguard (0 = off, raw dynamics)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    tag = args.mode + (f"_ft{args.freeze_top}" if args.freeze_top else "")

    trace_rows, hook_rows, final_rows, align_rows, loss_rows = run_experiment(
        args.lams, args.depths, args.seeds, args.steps, args.log_every,
        args.clip, args.mode, args.freeze_top, device,
    )

    write_csv(os.path.join(RESULTS_DIR, f"td_{tag}_traces.csv"), trace_rows)
    write_csv(os.path.join(RESULTS_DIR, f"td_{tag}_hooks.csv"), hook_rows)
    write_csv(os.path.join(RESULTS_DIR, f"td_{tag}_final.csv"), final_rows)
    write_csv(os.path.join(RESULTS_DIR, f"td_{tag}_alignment.csv"), align_rows)
    write_csv(os.path.join(RESULTS_DIR, f"td_{tag}_loss.csv"), loss_rows)

    plot_results(hook_rows, final_rows, align_rows, loss_rows, args.lams,
                 args.depths, args.steps, tag)


if __name__ == "__main__":
    main()
