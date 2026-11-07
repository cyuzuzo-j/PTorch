"""Validation-accuracy curves for any NeurIPS experiment, driven by its YAML.

    python -m experiments.neurips.plot_curves --config experiments/neurips/mlp/config.yaml

Each entry of the YAML's `plots:` list makes one figure in `figures_dir`:

    file:         output file name (required)
    x:            step | time       (time = training wall-clock, eval excluded)
    task:         only rows of this task          (default: all)
    where:        extra column filters, e.g. {batch_size: 32}; a list keeps any of its values
    title, legend_title, xlim, ylim, xscale
    time_bins:    bins used to average runs over wall-clock time (default 100)

One curve per variant, labelled through the YAML's `labels:` map, showing the
mean over seeds with a ±1 std band.
"""
import argparse
import os

import pandas as pd
import seaborn as sns

from experiments.neurips import common, style

style.apply_theme()


def plot(df, spec, out_path):
    if spec.get("task"):
        df = df[df["task"] == spec["task"]]
    for col, val in spec.get("where", {}).items():
        df = df[df[col].isin(val if isinstance(val, list) else [val])]
    if df.empty:
        print(f"  no data for {spec['file']}, skipping")
        return

    x = "step"
    if spec["x"] == "time":
        # Runs evaluate at different wall-clock times: bin them to average.
        df = df.copy()
        bins = pd.cut(df["wall_time_s"], bins=spec.get("time_bins", 100))
        df["time"] = bins.apply(lambda iv: iv.mid).astype(float)
        x = "time"

    labels = sorted(df["Label"].unique())
    # Colour by variant so a method looks the same in every figure.
    variants = df.drop_duplicates("Label").set_index("Label")["variant"]
    colors = style.color_map(sorted(variants))
    fig, ax = style.subplots()
    sns.lineplot(data=df, x=x, y="acc", hue="Label", hue_order=labels,
                 palette={l: colors[variants[l]] for l in labels},
                 estimator="mean", errorbar=("sd", 1), ax=ax)
    ax.set_xlabel("Optimization Step" if x == "step" else "Wall-clock Time (s)")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title(spec.get("title", ""))
    ax.legend(title=spec.get("legend_title"))
    if "xscale" in spec:
        ax.set_xscale(spec["xscale"])
    if "xlim" in spec:
        ax.set_xlim(*spec["xlim"])
    if "ylim" in spec:
        ax.set_ylim(*spec["ylim"])
    style.savefig(fig, out_path)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, help="experiment YAML")
    cfg = common.load_config(p.parse_args().config)
    df = common.load_runs(cfg, split="val")
    if df.empty:
        raise SystemExit(f"no results in {cfg['results_dir']}")
    os.makedirs(cfg["figures_dir"], exist_ok=True)
    for spec in cfg["plots"]:
        plot(df, spec, os.path.join(cfg["figures_dir"], spec["file"]))


if __name__ == "__main__":
    main()
