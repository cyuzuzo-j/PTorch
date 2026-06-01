"""
Plot results of the non-differentiable activations benchmark.

Reads the per-activation CSVs produced by quantized_relu.py and emits a
single figure per task that compares validation-accuracy training curves
across ReLU (differentiable) and Step / GappedStep / QuantizedRelu
(non-differentiable, no chain-rule derivative — trained via per-activation
projections in ptorch).
"""

import os
import sys
import glob
import argparse
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

sns.set_theme(style="whitegrid", context="paper", font_scale=1.6)

# Activation metadata: pretty label, plot color, differentiable?, line style.
ACTIVATION_META = {
    "ReLU":          {"label": "ReLU",          "color": "#3b528b", "diff": True,  "ls": "-"},
    "Step":          {"label": "Step",                            "color": "#21918c", "diff": False, "ls": "--"},
    "GappedStep":    {"label": "Gapped Step",                     "color": "#5ec962", "diff": False, "ls": "-."},
    "QuantizedRelu": {"label": "Quantized ReLU",                  "color": "#fde725", "diff": False, "ls": ":"},
    "Sort":          {"label": "Sort",                            "color": "#c2356d", "diff": False, "ls": (0, (3, 1, 1, 1))},
}


def load_results(results_dir: str, task_name: str) -> pd.DataFrame:
    pattern = os.path.join(results_dir, f"*{task_name}.csv")
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
    return pd.concat(frames, ignore_index=True)


def plot_task(df: pd.DataFrame, task_name: str, out_dir: str):
    if df.empty:
        print(f"  ! no data for {task_name}; skipping.")
        return

    # Drop the final 'test' bookkeeping row (it duplicates the last step).
    if "split" in df.columns:
        train_df = df[df["split"].isna() | (df["split"] != "test")].copy()
    else:
        train_df = df.copy()
    train_df = train_df.dropna(subset=["step", "val_acc"]).copy()
    train_df["step"] = train_df["step"].astype(int)

    fig, ax = plt.subplots(figsize=(14, 5))

    chance = 1.0 / 10  # MNIST/CIFAR10 are 10-class
    ax.axhline(chance, color="grey", lw=1, ls=":", label=f"chance ({chance:.1f})")

    # Preserve the canonical order rather than alphabetical.
    activation_order = [a for a in ACTIVATION_META if a in train_df["activation"].unique()]

    for activation in activation_order:
        sub = train_df[train_df["activation"] == activation]
        if sub.empty:
            continue
        meta = ACTIVATION_META[activation]
        # average across runs at each step
        agg = sub.groupby("step")["val_acc"].agg(["mean", "std", "count"]).reset_index()
        ax.plot(agg["step"], agg["mean"],
                color=meta["color"], ls=meta["ls"], lw=2.2,
                label=meta["label"])
        if (agg["count"] > 1).any():
            ax.fill_between(agg["step"],
                            agg["mean"] - agg["std"].fillna(0),
                            agg["mean"] + agg["std"].fillna(0),
                            color=meta["color"], alpha=0.15)

    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"{task_name}: non-differentiable activations train via projections")
    ax.legend(loc="lower right", fontsize=11, frameon=True)
    plt.tight_layout()

    pdf_path = os.path.join(out_dir, f"non_diff_activations_{task_name}.pdf")
    png_path = os.path.join(out_dir, f"non_diff_activations_{task_name}.png")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {pdf_path}")
    print(f"  saved {png_path}")


def main():
    here = os.path.dirname(__file__)
    parser = argparse.ArgumentParser(description="Plot non-differentiable activations benchmark")
    parser.add_argument("--config", default=os.path.join(here, "config.yaml"))
    parser.add_argument("--results-dir", default=None,
                        help="Override results dir (defaults to cfg['results_dir']).")
    parser.add_argument("--out-dir", default=os.path.join(here, "plots"))
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    results_dir = args.results_dir or os.path.join(here, cfg.get("results_dir", "results"))
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    tasks = [t["name"] for t in cfg["tasks"]]
    print(f"Tasks       : {tasks}")
    print(f"Results dir : {results_dir}")
    print(f"Output dir  : {out_dir}")

    for task_name in tasks:
        print(f"\n── {task_name} ──")
        df = load_results(results_dir, task_name)
        if df.empty:
            print(f"  ! no CSVs matching *{task_name}.csv in {results_dir}")
            continue
        plot_task(df, task_name, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
