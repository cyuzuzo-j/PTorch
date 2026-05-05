"""
Plot Deep MLP Benchmark Results
================================
Reads CSV logs produced by bench_ptorch.py and bench_torch.py for the
deep_mlp_config.yaml experiment, then produces two publication-quality figures:

  deep_mlp_STEP.png — Validation accuracy vs. optimization step
  deep_mlp_TIME.png — Validation accuracy vs. wall-clock time

The conditions compared are:
  PTorch 1-Layer [512]           — projection-based, shallow
  PTorch 4-Layer [512×4]         — projection-based, deep
  Torch 1-Layer [512]  (Adam)    — standard PyTorch baseline, shallow
  Torch 4-Layer [512×4] (Adam)   — standard PyTorch baseline, deep

Uses the same seaborn style as the other experiment plotters.
"""

import os
import sys
import glob
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# ── Seaborn theme — identical to the other plotters ──────────────────────────
sns.set_theme(style="whitegrid", context="paper", font_scale=2.0)

CFG_PATH = os.path.join(os.path.dirname(__file__), "deep_mlp_config.yaml")

# Curated palette — one color per condition
CONDITION_COLORS = {
    r"$\mathcal{P}$Torch 1-Layer [512]":    sns.color_palette("viridis", 4)[0],
    r"$\mathcal{P}$Torch 4-Layer [512×4]":  sns.color_palette("viridis", 4)[1],
    "Torch 1-Layer [512]":                   sns.color_palette("viridis", 4)[2],
    "Torch 4-Layer [512×4]":                 sns.color_palette("viridis", 4)[3],
}

DEPTH_LABELS = {
    "MNIST_1L": "1-Layer [512]",
    "MNIST_4L": "4-Layer [512×4]",
}


def load_all_results(results_dir: str, task_names: list[str]) -> pd.DataFrame:
    """Load and concatenate CSVs for all tasks, tagging each row with a condition label."""
    frames = []
    for task_name in task_names:
        pattern = os.path.join(results_dir, f"*_{task_name}.csv")
        for path in sorted(glob.glob(pattern)):
            df = pd.read_csv(path)
            df["task"] = task_name
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    depth_map = DEPTH_LABELS

    def label_condition(row):
        fw = row.get("framework", "")
        depth = depth_map.get(row.get("task", ""), row.get("task", ""))
        if fw == "ptorch":
            return rf"$\mathcal{{P}}$Torch {depth}"
        return f"Torch {depth}"

    df["Condition"] = df.apply(label_condition, axis=1)
    return df


def plot_figures(df: pd.DataFrame, output_dir: str):
    """Produce accuracy-vs-step and accuracy-vs-time figures."""
    if df.empty:
        print("  ⚠  No data found, skipping.")
        return

    # ── (a) Accuracy vs. Step ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.lineplot(
        data=df, x="step", y="val_acc",
        hue="Condition", estimator="mean", errorbar=("ci", 95),
        palette=CONDITION_COLORS, linewidth=2, ax=ax,
    )
    ax.set_xlabel("Optimization Step")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("MNIST MLP Depth — Accuracy vs. Step")
    ax.legend(loc="lower right", fontsize=10, frameon=False)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "deep_mlp_STEP.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")

    # ── (b) Accuracy vs. Wall-Clock Time ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))

    time_data = df.dropna(subset=["wall_time_s", "val_acc"]).copy()
    if not time_data.empty:
        n_bins = 50
        time_data["time_bin"] = pd.cut(time_data["wall_time_s"], bins=n_bins)
        time_data["time_mid"] = time_data["time_bin"].apply(
            lambda iv: iv.mid if iv is not None else None
        ).astype(float)

        sns.lineplot(
            data=time_data, x="time_mid", y="val_acc",
            hue="Condition", estimator="mean", errorbar=("ci", 95),
            palette=CONDITION_COLORS, linewidth=2, ax=ax,
        )

    ax.set_xlabel("Wall-clock Time (s)")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("MNIST MLP Depth — Accuracy vs. Time")
    ax.legend(loc="lower right", fontsize=10, frameon=False)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "deep_mlp_TIME.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    cfg = yaml.safe_load(open(CFG_PATH))
    results_dir = os.path.join(os.path.dirname(__file__), cfg.get("results_dir", "results/deep_mlp"))
    output_dir  = os.path.join(os.path.dirname(__file__), cfg.get("figures_dir", "../../images/mlp"))
    os.makedirs(output_dir, exist_ok=True)

    task_names = [t["name"] for t in cfg["tasks"]]
    print(f"Plotting deep MLP results for tasks: {task_names}")
    print(f"  Results dir : {results_dir}")
    print(f"  Output dir  : {output_dir}")

    df = load_all_results(results_dir, task_names)
    plot_figures(df, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
