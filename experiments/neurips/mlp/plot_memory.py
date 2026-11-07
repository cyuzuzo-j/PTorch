"""Peak memory vs input dimension or batch size (paper fig:memory), from bench_memory's CSV.

    python -m experiments.neurips.mlp.plot_memory --config experiments/neurips/mlp/memory_config.yaml
    python -m experiments.neurips.mlp.plot_memory --config experiments/neurips/mlp/memory_batch_config.yaml
"""
import argparse
import os

import pandas as pd

from experiments.neurips import common, style

style.apply_theme()

LABELS = {
    "ptorch": r"$\mathcal{P}$Torch",
    "ptorch_l2": r"$\mathcal{P}$Torch ($\ell_2$, fused)",
    "ptorch_linf": r"$\mathcal{P}$Torch ($\ell_\infty$)",
    "torch": "Torch",
    "pjax": "PJAX",
}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, help="experiment YAML")
    cfg = common.load_config(p.parse_args().config)
    batch = cfg.get("sweep", "in_dim") == "batch"
    name = "memory_vs_batch" if batch else "memory_vs_k"
    df = pd.read_csv(os.path.join(cfg["results_dir"], f"{name}.csv"))
    if not batch and "plot_max_k" in cfg:
        df = df[df["K"] <= cfg["plot_max_k"]]
    x = "B" if batch else "K"

    fig, ax = style.subplots()
    for fw in cfg["frameworks"]:
        col = f"{fw}_{cfg['metric']}_mb"
        if col not in df.columns or df[col].isna().all():
            continue
        sub = df.dropna(subset=[col])
        color = style.VARIANT_COLORS[fw]
        ax.plot(sub[x], sub[col], "o-", label=LABELS[fw], color=color)
        # A framework that ran out of memory: mark the first size it failed at.
        failed = df[df[col].isna() & (df[x] > sub[x].max())]
        if not failed.empty:
            ax.plot(failed[x].iloc[0], sub[col].iloc[-1], "x", color=color, ms=9, mew=2)
    if batch:
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("Batch Size")
        ax.set_title("Peak Memory Usage vs. Batch Size")
        ax.legend(loc="upper left")
    else:
        ax.set_xlabel("Input Dimension")
        ax.set_title("Peak Memory Usage vs. Input Neurons")
        ax.legend(loc="upper left")
    ax.set_ylabel("Peak Memory (MB)")

    os.makedirs(cfg["figures_dir"], exist_ok=True)
    style.savefig(fig, os.path.join(cfg["figures_dir"], f"{name}.pdf"))


if __name__ == "__main__":
    main()
