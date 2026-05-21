"""
Hybrid CNN training — comparison plotter
=========================================
Loads CSVs produced by bench_hybrid.py, bench_ptorch.py, and bench_torch.py,
then produces per-task figures with four panels each:

  (a) Validation accuracy vs. step
  (b) Validation accuracy vs. wall-clock time

For hybrid runs the gradient / projection phase boundaries are shaded on
the step-axis panel.

Output (one PDF per task):
  results/hybrid_MNIST.pdf
  results/hybrid_CIFAR10.pdf

Usage:
  python experiments/cnn_benchmarks/plot_hybrid.py [--results_dir PATH] [--out_dir PATH]
"""
import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

sns.set_theme(style="whitegrid", context="paper", font_scale=1.6)

HERE = os.path.dirname(os.path.abspath(__file__))

# ── Label & colour maps ───────────────────────────────────────────────────────

def _framework_label(row):
    fw = row.get("framework", "")
    if fw == "torch":
        opt = row.get("optimizer", "")
        return f"Torch ({opt})" if opt else "Torch"
    if fw in ("ptorch_cnn", "ptorch_cyclic_cnn"):
        opt = row.get("optimizer", "")
        return f"ProjTorch ({opt})" if opt else "ProjTorch"
    if fw == "hybrid_cnn":
        K        = row.get("K", "?")
        grad_opt = row.get("grad_opt", "")
        proj_opt = row.get("proj_opt", "")
        return f"Hybrid K={K} ({grad_opt}↔{proj_opt})"
    return fw


_PALETTE = sns.color_palette("tab10", 10)
_COLOR_BY_PREFIX = {
    "Torch":     _PALETTE[0],
    "ProjTorch": _PALETTE[1],
    "Hybrid":    _PALETTE[2],
}

def _color(label: str):
    for prefix, c in _COLOR_BY_PREFIX.items():
        if label.startswith(prefix):
            return c
    return "grey"


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_task(results_dir: str, task_name: str) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(results_dir, f"*_{task_name}.csv"))):
        df = pd.read_csv(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)

    # drop test-only sentinel rows
    if "phase" in df.columns:
        df = df[df["phase"] != "test"]

    # build a human-readable label for each row
    df["label"] = df.apply(_framework_label, axis=1)
    return df


# ── Phase-boundary shading (hybrid only) ────────────────────────────────────

def _shade_phases(ax, hybrid_df: pd.DataFrame, x_col: str):
    """Add alternating light-coloured bands for grad / proj phases."""
    if hybrid_df.empty or "phase" not in hybrid_df.columns:
        return

    # Use one representative run to infer phase boundaries
    run_ids = hybrid_df["run"].unique()
    ref = hybrid_df[hybrid_df["run"] == run_ids[0]].sort_values(x_col)

    phase_colors = {"grad": "#d4e6f1", "proj": "#d5f5e3"}
    prev_phase, prev_x = None, None

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    for _, r in ref.iterrows():
        phase = r["phase"]
        x_val = r[x_col]
        if prev_phase is not None and prev_phase != phase:
            ax.axvspan(prev_x, x_val,
                       color=phase_colors.get(prev_phase, "#eeeeee"),
                       alpha=0.25, linewidth=0, zorder=0)
        prev_phase = phase
        prev_x = x_val

    # shade the final segment up to axis limit
    if prev_phase is not None:
        ax.axvspan(prev_x, xlim[1],
                   color=phase_colors.get(prev_phase, "#eeeeee"),
                   alpha=0.25, linewidth=0, zorder=0)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # legend patches for phases
    handles = [
        mpatches.Patch(facecolor="#d4e6f1", alpha=0.5, label="grad phase"),
        mpatches.Patch(facecolor="#d5f5e3", alpha=0.5, label="proj phase"),
    ]
    return handles


# ── Per-task figure ───────────────────────────────────────────────────────────

def plot_task(df: pd.DataFrame, task_name: str, out_dir: str):
    if df.empty:
        print(f"  No data for {task_name}, skipping.")
        return

    labels  = sorted(df["label"].unique())
    palette = {lb: _color(lb) for lb in labels}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{task_name} — Hybrid vs ProjTorch vs Torch", fontsize=14)

    # ── (a) Accuracy vs. step ────────────────────────────────────────────────
    ax = axes[0]
    sns.lineplot(
        data=df, x="step", y="val_acc",
        hue="label", palette=palette,
        estimator="mean", errorbar=("ci", 95),
        linewidth=2, ax=ax,
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Accuracy vs. Step")

    hybrid_df = df[df["label"].str.startswith("Hybrid")]
    phase_handles = _shade_phases(ax, hybrid_df, "step")

    # Merge phase legend patches into the existing legend
    existing_handles, existing_labels = ax.get_legend_handles_labels()
    if phase_handles:
        ax.legend(handles=existing_handles + phase_handles,
                  labels=existing_labels + ["grad phase", "proj phase"],
                  loc="lower right", fontsize=10, frameon=False)
    else:
        ax.legend(loc="lower right", fontsize=10, frameon=False)

    # ── (b) Accuracy vs. wall-clock time ─────────────────────────────────────
    ax = axes[1]

    time_df = df.dropna(subset=["wall_time_s", "val_acc"]).copy()
    if not time_df.empty:
        n_bins = 60
        t_min  = time_df["wall_time_s"].min()
        t_max  = time_df["wall_time_s"].max()
        bins   = np.linspace(t_min, t_max, n_bins + 1)
        time_df["time_bin"] = pd.cut(time_df["wall_time_s"], bins=bins)
        time_df["time_mid"] = time_df["time_bin"].apply(
            lambda iv: float(iv.mid) if pd.notna(iv) else np.nan
        )
        time_df = time_df.dropna(subset=["time_mid"])

        sns.lineplot(
            data=time_df, x="time_mid", y="val_acc",
            hue="label", palette=palette,
            estimator="mean", errorbar=("ci", 95),
            linewidth=2, ax=ax,
        )

        hybrid_time = time_df[time_df["label"].str.startswith("Hybrid")]
        _shade_phases(ax, hybrid_time, "time_mid")

    ax.set_xlabel("Wall-clock Time (s)")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Accuracy vs. Wall-clock Time")
    ax.legend(loc="lower right", fontsize=10, frameon=False)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"hybrid_{task_name}.pdf")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir",
                        default=os.path.join(HERE, "results"))
    parser.add_argument("--out_dir",
                        default=os.path.join(HERE, "results"))
    parser.add_argument("--tasks", nargs="+",
                        default=["MNIST", "CIFAR10"])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Results dir : {args.results_dir}")
    print(f"Output dir  : {args.out_dir}")

    for task in args.tasks:
        print(f"\n── {task} ──")
        df = _load_task(args.results_dir, task)
        plot_task(df, task, args.out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
