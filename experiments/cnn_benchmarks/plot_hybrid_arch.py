"""
Plot results of the hybrid-architecture CNN benchmark.

Reads the per-framework CSVs produced by bench_ptorch_hybrid_arch.py,
bench_ptorch.py, and bench_torch.py, then emits a single figure per task
that compares validation-accuracy training curves across Torch (chain-rule
SGD), $\\mathcal{P}$Torch (pure projection optimizer on the whole net),
and Hybrid (plain torch SGD on the conv backbone + ptorch projection on
the head).
"""

import os
import sys
import glob
import argparse

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=1.6)

HERE = os.path.dirname(os.path.abspath(__file__))

# Framework metadata: pretty label, plot color, line style.
FRAMEWORK_META = {
    "torch":                  {"label": "Torch",                     "color": sns.color_palette("viridis", 3)[0], "ls": "--"},
    "ptorch_cnn":             {"label": r"$\mathcal{P}$Torch",       "color": sns.color_palette("viridis", 3)[1], "ls": "-"},
    "ptorch_cnn_hybrid_arch": {"label": r"Hybrid ($\mathcal{P}$Torch head)", "color": sns.color_palette("viridis", 3)[2], "ls": "-"},
}


def load_results(results_dir: str, task_name: str) -> pd.DataFrame:
    pattern = os.path.join(results_dir, f"*_{task_name}.csv")
    frames = []
    for path in sorted(glob.glob(pattern)):
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"  ! failed to read {path}: {exc}")
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df[df["framework"].isin(FRAMEWORK_META)].copy()
    return df


def plot_task(df: pd.DataFrame, task_name: str, out_dir: str):
    if df.empty:
        print(f"  ! no data for {task_name}; skipping.")
        return

    if "split" in df.columns:
        train_df = df[df["split"].isna() | (df["split"] != "test")].copy()
    else:
        train_df = df.copy()
    train_df = train_df.dropna(subset=["step", "val_acc"]).copy()
    train_df["step"] = train_df["step"].astype(int)

    fig, ax = plt.subplots(figsize=(14, 5))

    chance = 1.0 / 10  # MNIST/CIFAR10 are 10-class
    ax.axhline(chance, color="grey", lw=1, ls=":", label=f"chance ({chance:.1f})")

    # Preserve canonical framework order rather than alphabetical.
    framework_order = [f for f in FRAMEWORK_META if f in train_df["framework"].unique()]

    for framework in framework_order:
        sub = train_df[train_df["framework"] == framework]
        if sub.empty:
            continue
        meta = FRAMEWORK_META[framework]
        agg = sub.groupby("step")["val_acc"].agg(["mean", "std", "count"]).reset_index()
        ax.plot(agg["step"], agg["mean"],
                color=meta["color"], ls=meta["ls"], lw=2,
                label=meta["label"])
        if (agg["count"] > 1).any():
            ax.fill_between(agg["step"],
                            agg["mean"] - agg["std"].fillna(0),
                            agg["mean"] + agg["std"].fillna(0),
                            color=meta["color"], alpha=0.15)

    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(rf"{task_name}: hybrid conv backbone + $\mathcal{{P}}$Torch head")
    ax.legend(loc="lower right", fontsize=11, frameon=True)
    plt.tight_layout()

    pdf_path = os.path.join(out_dir, f"hybrid_arch_{task_name}.pdf")
    png_path = os.path.join(out_dir, f"hybrid_arch_{task_name}.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {pdf_path}")
    print(f"  saved {png_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot hybrid-architecture CNN benchmark")
    parser.add_argument("--results_dir", default=os.path.join(HERE, "results"))
    parser.add_argument("--out_dir", default=os.path.join(HERE, "results"))
    parser.add_argument("--tasks", nargs="+", default=["CIFAR10"])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Tasks       : {args.tasks}")
    print(f"Results dir : {args.results_dir}")
    print(f"Output dir  : {args.out_dir}")

    for task_name in args.tasks:
        print(f"\n── {task_name} ──")
        df = load_results(args.results_dir, task_name)
        if df.empty:
            print(f"  ! no CSVs matching *_{task_name}.csv in {args.results_dir}")
            continue
        plot_task(df, task_name, args.out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
