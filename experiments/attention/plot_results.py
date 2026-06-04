"""
Plot Attention Benchmark Results
================================
Reads all CSV logs under `results/` (both the ptorch projection runs and the
vanilla torch baseline) and produces:

    plots/attention_<task>_STEP.pdf — accuracy vs. optimization step
    plots/attention_<task>_TIME.pdf — accuracy vs. wall-clock time
"""

import os
import glob
import argparse

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns


sns.set_theme(style="whitegrid", context="paper", font_scale=2.0)


def _label(row):
    """Build a legend label for a row, distinguishing torch vs ptorch."""
    opt = row.get("optimizer", "?")
    loss = row.get("loss", "?")
    if row["framework"] == "torch":
        return f"Torch ({opt}, {loss})"
    norm = row.get("norm", "l2")
    return r"$\mathcal{P}$Torch" + f" ({norm}, {opt}, {loss})"


def _load(results_dir):
    files = glob.glob(os.path.join(results_dir, "*.csv"))
    if not files:
        print(f"  ⚠  No CSV files found in {results_dir}")
        return None

    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except Exception as e:
            print(f"  ⚠  Error reading {f}: {e}")

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    df = df[df["step"] >= 0].copy()
    df["Framework"] = df.apply(_label, axis=1)
    return df


def _plot_one(df, x_col, x_label, title, out_path, has_train, color_map,
              n_time_bins=50):
    """Plot val (solid) + train (dashed) accuracy against x_col."""
    fig, ax = plt.subplots(figsize=(14, 5))

    plot_df = df.dropna(subset=[x_col, "val_acc"]).copy()
    if plot_df.empty:
        plt.close(fig)
        return

    # Bin wall-clock time onto a uniform grid so multiple runs average cleanly.
    if x_col == "wall_time_s":
        plot_df["x_mid"] = (
            pd.cut(plot_df[x_col], bins=n_time_bins)
              .apply(lambda iv: iv.mid if iv is not None else None)
              .astype(float)
        )
        x_plot = "x_mid"
    else:
        x_plot = x_col

    sns.lineplot(
        data=plot_df, x=x_plot, y="val_acc",
        hue="Framework", estimator="mean", errorbar=("ci", 95),
        palette=color_map, linewidth=2, ax=ax,
    )

    train_plotted = False
    if has_train and plot_df["train_acc"].notna().any():
        sns.lineplot(
            data=plot_df, x=x_plot, y="train_acc",
            hue="Framework", estimator="mean", errorbar=None,
            palette=color_map, linewidth=2, linestyle="--",
            alpha=0.8, ax=ax, legend=False,
        )
        train_plotted = True

    ax.set_xlabel(x_label)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.tick_params(axis="both", labelsize=12)
    ax.xaxis.get_offset_text().set_fontsize(12)

    framework_legend = ax.legend(loc="lower right", fontsize=11, frameon=False)

    if train_plotted:
        style_handles = [
            Line2D([0], [0], color="black", linewidth=2, linestyle="-",  label="Val"),
            Line2D([0], [0], color="black", linewidth=2, linestyle="--", label="Train"),
        ]
        ax.add_artist(framework_legend)
        ax.legend(handles=style_handles, loc="upper left",
                  fontsize=11, frameon=False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_attention_results(results_dir="results", output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    df = _load(results_dir)
    if df is None:
        return

    has_train = "train_acc" in df.columns and df["train_acc"].notna().any()

    # Stable color per Framework across all subplots / both axes.
    frameworks = sorted(df["Framework"].unique())
    palette = sns.color_palette("viridis", max(len(frameworks), 2))
    color_map = dict(zip(frameworks, palette))

    tasks = sorted(df["task"].dropna().unique())
    for task in tasks:
        task_df = df[df["task"] == task]
        _plot_one(
            task_df, "step", "Optimization Step",
            f"{task} — Accuracy vs. Step",
            os.path.join(output_dir, f"attention_{task}_STEP.pdf"),
            has_train, color_map,
        )
        _plot_one(
            task_df, "wall_time_s", "Wall-clock Time (s)",
            f"{task} — Accuracy vs. Time",
            os.path.join(output_dir, f"attention_{task}_TIME.pdf"),
            has_train, color_map,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results",
                        help="Directory containing CSV results")
    parser.add_argument("--output", default="plots",
                        help="Directory to save plots")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    plot_attention_results(args.results, args.output)
    print("\nDone.")
