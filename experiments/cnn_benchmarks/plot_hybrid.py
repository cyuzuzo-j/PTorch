"""
Hybrid CNN training — comparison plotter
=========================================
Loads CSVs produced by bench_hybrid.py, bench_ptorch.py, and bench_torch.py,
then produces a single figure with one panel per task (CIFAR10 and MNIST),
each plotting validation accuracy vs. step. Phase boundaries are shaded.

Output:
  results/hybrid_step.pdf

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
        return rf"$\mathcal{{P}}$Torch ({opt})" if opt else r"$\mathcal{P}$Torch"
    if fw == "hybrid_cnn":
        K        = row.get("K", "?")
        grad_opt = row.get("grad_opt", "")
        proj_opt = row.get("proj_opt", "")
        return f"Hybrid K={K}"
    return fw


# Frameworks that should never appear in this plot.
_EXCLUDED_FRAMEWORKS = {"ptorch_cnn_hybrid_arch"}


def _viridis_palette(labels):
    """Map labels to viridis colors, ordered for stable comparison."""
    labels = list(labels)
    if not labels:
        return {}
    colors = sns.color_palette("viridis", max(len(labels), 2))
    return {lb: colors[i] for i, lb in enumerate(labels)}


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_task(results_dir: str, task_name: str) -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(results_dir, f"*_{task_name}.csv"))):
        df = pd.read_csv(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)

    # drop frameworks we don't want to show here (e.g. hybrid-arch variants)
    if "framework" in df.columns:
        df = df[~df["framework"].isin(_EXCLUDED_FRAMEWORKS)]

    # drop test-only sentinel rows
    if "phase" in df.columns:
        df = df[df["phase"] != "test"]

    # build a human-readable label for each row
    df["label"] = df.apply(_framework_label, axis=1)
    return df


# ── Phase shading (hybrid only) ─────────────────────────────────────────────
#
# Minimalist: only the projection phase is tinted with a pale viridis hue.
# The gradient phase is left as clean whitespace — the alternation reads as
# light-on-blank bands without competing with the curves.

_PROJ_COLOR = plt.get_cmap("viridis")(0.75)   # warm yellow-green
_PROJ_BAND_ALPHA  = 0.08
_PROJ_PATCH_ALPHA = 0.35


def _phase_intervals(ref_df: pd.DataFrame, x_col: str):
    """Return [(start_x, end_x, phase), ...] inferred from one reference run."""
    if ref_df.empty:
        return []
    ref_df = ref_df.sort_values(x_col).reset_index(drop=True)

    intervals = []
    start_x = ref_df.iloc[0][x_col]
    cur     = ref_df.iloc[0]["phase"]

    for i in range(1, len(ref_df)):
        ph = ref_df.iloc[i]["phase"]
        x  = ref_df.iloc[i][x_col]
        if ph != cur:
            mid = 0.5 * (ref_df.iloc[i - 1][x_col] + x)
            intervals.append((start_x, mid, cur))
            start_x = mid
            cur = ph
    intervals.append((start_x, ref_df.iloc[-1][x_col], cur))
    return intervals


def _shade_phases(ax, hybrid_df: pd.DataFrame, x_col: str):
    """Tint only the projection phase with a pale viridis band."""
    if hybrid_df.empty or "phase" not in hybrid_df.columns:
        return None

    run_ids = hybrid_df["run"].unique()
    ref = hybrid_df[hybrid_df["run"] == run_ids[0]]
    intervals = _phase_intervals(ref, x_col)
    if not intervals:
        return None

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    shaded_any = False
    for start_x, end_x, phase in intervals:
        if phase != "proj":
            continue
        ax.axvspan(start_x, end_x,
                   facecolor=_PROJ_COLOR, alpha=_PROJ_BAND_ALPHA,
                   linewidth=0, zorder=0)
        shaded_any = True

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if not shaded_any:
        return None
    return [mpatches.Patch(facecolor=_PROJ_COLOR,
                           alpha=_PROJ_PATCH_ALPHA,
                           edgecolor="none",
                           label="projection step")]


# ── Per-task panel ────────────────────────────────────────────────────────────

def _plot_step_panel(ax, df: pd.DataFrame, task_name: str):
    if df.empty:
        ax.set_title(f"{task_name} — no data")
        ax.set_xlabel("Step")
        ax.set_ylabel("Validation Accuracy")
        return

    labels  = sorted(df["label"].unique())
    palette = _viridis_palette(labels)

    sns.lineplot(
        data=df, x="step", y="val_acc",
        hue="label", hue_order=labels, palette=palette,
        estimator="mean", errorbar=("ci", 95),
        linewidth=2, ax=ax,
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title(task_name)

    # MNIST converges near the ceiling — zoom in so the curves are readable.
    if task_name == "MNIST":
        ax.set_ylim(0.9, 1.0)
    ax.set_xlim(0, 4800)

    hybrid_df = df[df["label"].str.startswith("Hybrid")]
    phase_handles = _shade_phases(ax, hybrid_df, "step")

    existing_handles, existing_labels = ax.get_legend_handles_labels()
    if phase_handles:
        phase_labels = [h.get_label() for h in phase_handles]
        ax.legend(handles=existing_handles + phase_handles,
                  labels=existing_labels + phase_labels,
                  loc="lower right", fontsize=10, frameon=False)
    else:
        ax.legend(loc="lower right", fontsize=10, frameon=False)


def plot_tasks(dfs: dict, out_dir: str):
    """dfs: {task_name: dataframe}. Renders one panel per task on a single figure."""
    tasks = list(dfs.keys())
    fig, axes = plt.subplots(1, len(tasks), figsize=(7 * len(tasks), 5), squeeze=False)
    fig.suptitle(r"Hybrid vs $\mathcal{P}$Torch vs Torch — Accuracy vs. Step", fontsize=14)
    for ax, task in zip(axes[0], tasks):
        _plot_step_panel(ax, dfs[task], task)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "hybrid_step.pdf")
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
                        default=["CIFAR10", "MNIST"])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Results dir : {args.results_dir}")
    print(f"Output dir  : {args.out_dir}")

    dfs = {}
    for task in args.tasks:
        print(f"\n── {task} ──")
        dfs[task] = _load_task(args.results_dir, task)
    plot_tasks(dfs, args.out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
