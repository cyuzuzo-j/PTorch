"""
plot_memory.py
==============
Plot peak memory vs input dim K from ``results/memory_vs_k.csv``.

Reproduces the layout of ``images/mlp/memory_vs_k_report.png`` using the
same seaborn style as ``plot_mlp_benchmark.py``.
"""

import argparse
import os
import os.path as osp

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid", context="paper", font_scale=2.0)


FRAMEWORK_LABELS = {
    "ptorch": r"$\mathcal{P}$Torch",
    "torch":  "Torch",
    "pjax":   "PJAX",
}

FRAMEWORK_COLORS = {
    r"$\mathcal{P}$Torch": sns.color_palette("viridis", 3)[0],
    "Torch":               sns.color_palette("viridis", 3)[1],
    "PJAX":                sns.color_palette("viridis", 3)[2],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=osp.join(osp.dirname(__file__),
                                              "results", "memory_vs_k.csv"))
    p.add_argument("--out", default=osp.join(osp.dirname(__file__),
                                              "..", "..", "images", "mlp",
                                              "memory_vs_k.pdf"))
    p.add_argument("--metric", choices=["rss", "tracemalloc"], default="rss")
    args = p.parse_args()

    os.makedirs(osp.dirname(osp.abspath(args.out)), exist_ok=True)

    df = pd.read_csv(args.csv)
    suffix = "_rss_mb" if args.metric == "rss" else "_tracemalloc_mb"
    M, B = int(df["M"].iloc[0]), int(df["B"].iloc[0])

    fig, ax = plt.subplots(figsize=(7, 5))
    for fw, label in FRAMEWORK_LABELS.items():
        col = f"{fw}{suffix}"
        if col not in df.columns or df[col].isna().all():
            continue
        sub = df.dropna(subset=[col])
        ax.plot(
            sub["K"], sub[col], "o-",
            label=label, color=FRAMEWORK_COLORS[label],
            linewidth=2, markersize=8,
        )

    ax.set_xlabel("Input dim ")
    ax.set_ylabel("Peak memory (MB)")
    ax.set_title(f"Peak memory vs input neurons (output dim={M}, batch size={B})")
    ax.legend(loc="lower right", fontsize=12, frameon=False)

    plt.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
