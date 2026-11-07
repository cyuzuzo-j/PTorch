"""PJAX / PTorch peak-memory ratio of the 1-block CNN as the conv layer grows, from bench_memory's CSV.
Sizes where PJAX ran out of GPU memory are marked with a cross at the top of the panel.

    python -m experiments.neurips.cnn_benchmarks.plot_memory --config experiments/neurips/cnn_benchmarks/memory_config.yaml
"""
import argparse
import os

import pandas as pd
from matplotlib.lines import Line2D

from experiments.neurips import common, style


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, help="experiment YAML")
    cfg = common.load_config(p.parse_args().config)
    style.apply_theme()

    df = pd.read_csv(os.path.join(cfg["results_dir"], "memory_vs_conv.csv"))
    ratio = df["pjax_device_mb"] / df["ptorch_device_mb"]
    ok = ratio.notna()
    x = df["conv_params"]
    top = ratio[ok].max() * 1.25

    fig, ax = style.subplots()
    ax.plot(x[ok], ratio[ok], "o-", color=style.NAVY, lw=2.5, ms=9)
    for xv, r in zip(x[ok], ratio[ok]):
        ax.annotate(f"{r:.0f}x", (xv, r), textcoords="offset points", xytext=(10, -4), ha="left", va="top",
                    fontsize=14)
    ax.plot(x[~ok], [top / 1.1] * (~ok).sum(), "x", color=style.NAVY, ms=12, mew=3)

    ax.set_xscale("log")
    ax.set_xlim(x.min() / 1.6, x.max() * 1.6)
    ax.set_xticks(x, [f"{k}x{k}\n{f}" for k, f in zip(df["kernel"], df["filters"])])
    ax.minorticks_off()
    ax.set_ylim(0, top)
    ax.set_xlabel("Conv kernel / filters")
    ax.set_ylabel(r"PJAX / $\mathcal{P}$Torch peak memory")
    ax.set_title(r"CNN Memory Scaling: PJAX vs. $\mathcal{P}$Torch")
    handles = [Line2D([], [], color=style.NAVY, marker="o", lw=2.5, ms=9, label=r"PJAX / $\mathcal{P}$Torch")]
    if (~ok).any():
        handles.append(Line2D([], [], color=style.NAVY, marker="x", ls="none", ms=12, mew=3,
                              label=f"PJAX out of memory ({cfg['gpu_memory_gb']} GB)"))
    ax.legend(handles=handles, loc="lower right", frameon=True)

    os.makedirs(cfg["figures_dir"], exist_ok=True)
    style.savefig(fig, os.path.join(cfg["figures_dir"], "memory_ratio_joint.pdf"))


if __name__ == "__main__":
    main()
