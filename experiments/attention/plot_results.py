"""
Plot Attention Benchmark Results
====================================
Reads CSV logs and produces publication-quality figures:

  attention_STEP.pdf — Accuracy vs. optimization step
  attention_TIME.pdf — Accuracy vs. wall-clock time
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# ── Seaborn theme ────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="paper", font_scale=2.0)

def plot_attention_results(results_dir: str = "results", output_dir: str = "plots"):
    os.makedirs(output_dir, exist_ok=True)

    # ── Load Data ────────────────────────────────────────────────────────────
    all_files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not all_files:
        print(f"  ⚠  No CSV files found in {results_dir}")
        return

    df_list = []
    for f in all_files:
        try:
            df_list.append(pd.read_csv(f))
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if not df_list:
        return

    df = pd.concat(df_list, ignore_index=True)
    df = df[df['step'] >= 0]

    def create_label(row):
        if row['framework'] == 'torch':
            return f"Torch (AdamW, {row['loss']})"
        norm = row.get('norm', 'l2')
        opt = row.get('optimizer', 'ProjMuon')
        loss = row.get('loss', 'CE')
        return r"$\mathcal{P}$Torch" + f" ({norm}, {opt}, {loss})"

    df['Framework'] = df.apply(create_label, axis=1)
    has_train = 'train_acc' in df.columns and df['train_acc'].notna().any()

    # Create a dynamic palette using seaborn's viridis mapped to unique frameworks
    unique_frameworks = sorted(df['Framework'].unique())
    palette = sns.color_palette("viridis", len(unique_frameworks))
    color_map = dict(zip(unique_frameworks, palette))

    # ── (a) Accuracy vs. Step ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Plot Validation Accuracy (Solid, Mean + CI)
    sns.lineplot(
        data=df, x="step", y="val_acc",
        hue="Framework", estimator="mean", errorbar=("ci", 95),
        palette=color_map, linewidth=2, ax=ax,
    )

    # Plot Train Accuracy (Dashed, Mean only to reduce visual clutter)
    if has_train:
        sns.lineplot(
            data=df, x="step", y="train_acc",
            hue="Framework", estimator="mean", errorbar=None,
            palette=color_map, linewidth=2, linestyle="--", alpha=0.8, ax=ax, legend=False
        )

    ax.set_xlabel("Optimization Step")
    ax.set_ylabel("Accuracy")
    ax.set_title(r"$\mathcal{P}$Torch Attention — Accuracy vs. Step")
    ax.legend(loc="lower right", fontsize=12, frameon=False)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "attention_STEP.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")

    # ── (b) Accuracy vs. Wall-Clock Time ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))

    # Bin unaligned wall-clock times into uniform intervals for clean averaging
    time_data = df.dropna(subset=["wall_time_s", "val_acc"]).copy()
    if not time_data.empty:
        n_bins = 50
        time_data["time_bin"] = pd.cut(time_data["wall_time_s"], bins=n_bins)
        time_data["time_mid"] = time_data["time_bin"].apply(
            lambda iv: iv.mid if iv is not None else None
        ).astype(float)

        sns.lineplot(
            data=time_data, x="time_mid", y="val_acc",
            hue="Framework", estimator="mean", errorbar=("ci", 95),
            palette=color_map, linewidth=2, ax=ax,
        )

        if has_train and "train_acc" in time_data.columns:
            sns.lineplot(
                data=time_data, x="time_mid", y="train_acc",
                hue="Framework", estimator="mean", errorbar=None,
                palette=color_map, linewidth=2, linestyle="--", alpha=0.8, ax=ax, legend=False
            )

    ax.set_xlabel("Wall-clock Time (s)")
    ax.set_ylabel("Accuracy")
    ax.set_title(r"$\mathcal{P}$Torch Attention — Accuracy vs. Time")
    ax.legend(loc="lower right", fontsize=12, frameon=False)

    plt.tight_layout()
    out_path_time = os.path.join(output_dir, "attention_TIME.pdf")
    fig.savefig(out_path_time, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results", help="Directory containing CSV results")
    parser.add_argument("--output", default="plots", help="Directory to save plots")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    plot_attention_results(args.results, args.output)
    print("\nDone.")