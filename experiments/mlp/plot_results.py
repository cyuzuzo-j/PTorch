"""
Plot Shallow MLP Benchmark Results
====================================
Reads CSV logs produced by bench_ptorch.py, bench_torch.py, and bench_pjax.py,
then produces four publication-quality figures:

  MNIST_1L_STEP.png   — Validation accuracy vs. optimization step (MNIST)
  MNIST_1L_TIME.png   — Validation accuracy vs. wall-clock time  (MNIST)
  CIFAR10_1L_STEP.png — Validation accuracy vs. optimization step (CIFAR-10)
  CIFAR10_1L_TIME.png — Validation accuracy vs. wall-clock time  (CIFAR-10)

Uses the same seaborn style as experiments/deep/vanishing_target.py.
"""

import os
import sys
import glob
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# ── Seaborn theme — identical to vanishing_target.py ─────────────────────────
sns.set_theme(style="whitegrid", context="paper", font_scale=2.0)

CFG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

# Pretty display names for the legend
FRAMEWORK_LABELS = {
    "ptorch": r"$\mathcal{P}$Torch",
    "torch":  "Torch",
    "pjax":   "PJAX",
}

# Curated palette (viridis-derived) — one color per framework
FRAMEWORK_COLORS = {
    r"$\mathcal{P}$Torch": sns.color_palette("viridis", 3)[0],
    "Torch":               sns.color_palette("viridis", 3)[1],
    "PJAX":                sns.color_palette("viridis", 3)[2],
}


def load_results(results_dir: str, task_name: str) -> pd.DataFrame:
    """Load and concatenate all framework CSVs for a given task."""
    frames = []
    pattern = os.path.join(results_dir, f"*_{task_name}.csv")
    for path in sorted(glob.glob(pattern)):
        df = pd.read_csv(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # Map framework names to pretty labels
    df["Framework"] = df["framework"].map(FRAMEWORK_LABELS).fillna(df["framework"])
    return df


def plot_task(df: pd.DataFrame, task_name: str, output_dir: str):
    """
    Produce two figures for a single task:
      (a) Validation accuracy vs. optimization step
      (b) Validation accuracy vs. wall-clock time
    """
    if df.empty:
        print(f"  ⚠  No data for {task_name}, skipping.")
        return

    dataset_label = task_name.split("_")[0]  # "MNIST" or "CIFAR10"

    # ── (a) Accuracy vs. Step ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.lineplot(
        data=df, x="step", y="val_acc",
        hue="Framework", estimator="mean", errorbar=("ci", 95),
        palette=FRAMEWORK_COLORS, linewidth=2, ax=ax,
    )
    ax.set_xlabel("Optimization Step")
    ax.set_ylabel("Validation Accuracy")

    ax.set_title(f"{dataset_label} — Accuracy vs. Step")
    ax.legend(loc="lower right", fontsize=12, frameon=False)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{task_name}_STEP.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")

    # ── (b) Accuracy vs. Wall-Clock Time ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))

    # Bin unaligned wall-clock times into uniform intervals for clean averaging
    time_data = df.dropna(subset=["wall_time_s", "val_acc"]).copy()
    if not time_data.empty:
        n_bins = 10000
        time_data["time_bin"] = pd.cut(time_data["wall_time_s"], bins=n_bins)
        time_data["time_mid"] = time_data["time_bin"].apply(
            lambda iv: iv.mid if iv is not None else None
        ).astype(float)

        sns.lineplot(
            data=time_data, x="time_mid", y="val_acc",
            hue="Framework", estimator="mean", errorbar=("ci", 95),
            palette=FRAMEWORK_COLORS, linewidth=2, ax=ax,
        )

    ax.set_xlabel("Wall-clock Time (s)")
    ax.set_ylabel("Validation Accuracy")
    ax.set_xscale("log")
    ax.set_title(f"{dataset_label} — Accuracy vs. Time")
    ax.legend(loc="lower right", fontsize=12, frameon=False)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{task_name}_TIME.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    cfg = yaml.safe_load(open(CFG_PATH))
    results_dir = os.path.join(os.path.dirname(__file__), cfg.get("results_dir", "results"))
    output_dir  = os.path.join(os.path.dirname(__file__), cfg.get("figures_dir", "../../images/mlp"))
    os.makedirs(output_dir, exist_ok=True)

    tasks = [t["name"] for t in cfg["tasks"]]
    print(f"Plotting results for tasks: {tasks}")
    print(f"  Results dir : {results_dir}")
    print(f"  Output dir  : {output_dir}")

    for task_name in tasks:
        print(f"\n── {task_name} ──")
        df = load_results(results_dir, task_name)
        plot_task(df, task_name, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
