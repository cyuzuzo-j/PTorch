"""
Plot Deep MLP Benchmark Results
================================
Reads CSV logs produced by bench_ptorch.py and bench_torch.py for the
deep_mlp_config.yaml experiment, then produces two publication-quality figures:

  deep_mlp_STEP.png — Validation accuracy vs. optimization step
  deep_mlp_TIME.png — Validation accuracy vs. wall-clock time

The conditions compared are:
  PTorch 1-Layer [512]   (with / without muon on activations)
  PTorch 4-Layer [512×4] (with / without muon on activations)
  Torch  1-Layer [512]   (Adam baseline)
  Torch  4-Layer [512×4] (Adam baseline)

Run 1 of each condition is dropped from the curves because it includes the
one-time compile / initialization cost; that init cost is reported separately
in the legend of the wall-clock figure.
"""

import os
import sys
import glob
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=2.0)

CFG_PATH = os.path.join(os.path.dirname(__file__), "deep_mlp_config.yaml")

DEPTH_LABELS = {
    "MNIST_1L": "1-Layer [512]",
    "MNIST_4L": "4-Layer [512×4]",
}


def _condition_label(fw: str, depth: str, use_muon_act: bool) -> str:
    if fw == "ptorch":
        base = rf"$\mathcal{{P}}$Torch {depth}"
        return base + (r" (ortho hidden-states)" if use_muon_act else "")
    return f"Torch {depth}"


def _build_palette(conditions):
    """Assign a stable color to each Condition string."""
    palette = sns.color_palette("viridis", max(len(conditions), 4))
    return {c: palette[i % len(palette)] for i, c in enumerate(sorted(conditions))}


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

    if "use_muon_activations" not in df.columns:
        df["use_muon_activations"] = False
    df["use_muon_activations"] = df["use_muon_activations"].fillna(False).astype(bool)

    def label_condition(row):
        fw = row.get("framework", "")
        depth = DEPTH_LABELS.get(row.get("task", ""), row.get("task", ""))
        return _condition_label(fw, depth, bool(row.get("use_muon_activations", False)))

    df["Condition"] = df.apply(label_condition, axis=1)
    return df


def compute_init_times(df: pd.DataFrame) -> dict:
    """For each Condition, estimate the one-time compile / init overhead as the
    difference between run-1's earliest wall-time and the mean earliest wall-time
    of subsequent runs."""
    init_times = {}
    for cond, sub in df.groupby("Condition"):
        first_step = sub["step"].min()
        step0 = sub[sub["step"] == first_step]
        run1 = step0[step0["run"] == 1]["wall_time_s"]
        rest = step0[step0["run"] != 1]["wall_time_s"]
        if run1.empty:
            init_times[cond] = 0.0
            continue
        baseline = float(rest.mean()) if not rest.empty else 0.0
        init_times[cond] = max(0.0, float(run1.iloc[0]) - baseline)
    return init_times


def plot_figures(df: pd.DataFrame, output_dir: str):
    """Produce accuracy-vs-step and accuracy-vs-time figures."""
    if df.empty:
        print("  ⚠  No data found, skipping.")
        return

    init_times = compute_init_times(df)

    # Drop run 1 — it includes one-time compile / initialization overhead.
    df = df[df["run"] != 1].copy()
    if df.empty:
        print("  ⚠  After dropping run 1, no data remains.")
        return

    conditions = sorted(df["Condition"].unique())
    palette = _build_palette(conditions)

    # ── (a) Accuracy vs. Step ────────────────────────────────────────────────
    def _cond_row(fw, task, muon):
        return _condition_label(fw, DEPTH_LABELS[task], muon)

    left_conditions = [
        _cond_row("torch",  "MNIST_4L", False),
        _cond_row("ptorch", "MNIST_1L", False),
        _cond_row("ptorch", "MNIST_4L", False),
    ]
    right_conditions = [
        _cond_row("torch",  "MNIST_4L", False),
        _cond_row("ptorch", "MNIST_1L", False),
        _cond_row("ptorch", "MNIST_4L", False),
        _cond_row("ptorch", "MNIST_4L", True),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, conds, title in (
        (axes[0], left_conditions,  "Baseline"),
        (axes[1], right_conditions, "Hidden-state orthogonalization"),
    ):
        sub = df[df["Condition"].isin(conds)]
        sns.lineplot(
            data=sub, x="step", y="val_acc",
            hue="Condition", hue_order=conds,
            estimator="mean", errorbar=("ci", 95),
            palette={c: palette[c] for c in conds},
            linewidth=2, ax=ax,
        )
        ax.set_xlabel("Optimization Step")
        ax.set_title(title)
        ax.set_ylim(0.8, 1.0)
        ax.legend(loc="lower right", fontsize=10, frameon=False, title="Model")

    axes[0].set_ylabel("Validation Accuracy")
    axes[1].set_ylabel("")
    fig.suptitle("MNIST MLP Depth — Accuracy vs. Step")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "deep_mlp_STEP.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")

    # ── (b) Accuracy vs. Wall-Clock Time ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))

    time_data = df.dropna(subset=["wall_time_s", "val_acc"]).copy()
    legend_labels = {c: f"{c}  (init {init_times.get(c, 0.0):.1f}s)" for c in conditions}

    if not time_data.empty:
        n_bins = 50
        time_data["time_bin"] = pd.cut(time_data["wall_time_s"], bins=n_bins)
        time_data["time_mid"] = time_data["time_bin"].apply(
            lambda iv: iv.mid if iv is not None else None
        ).astype(float)
        time_data["Condition"] = time_data["Condition"].map(legend_labels)

        relabeled_palette = {legend_labels[c]: palette[c] for c in conditions}
        relabeled_order = [legend_labels[c] for c in conditions]

        sns.lineplot(
            data=time_data, x="time_mid", y="val_acc",
            hue="Condition", hue_order=relabeled_order,
            estimator="mean", errorbar=("ci", 95),
            palette=relabeled_palette, linewidth=2, ax=ax,
        )

    ax.set_xlabel("Wall-clock Time (s, training only — run 1 dropped)")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("MNIST MLP Depth — Accuracy vs. Time")
    ax.legend(loc="lower right", fontsize=10, frameon=False)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "deep_mlp_TIME.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")

    print("\nCompile / init overhead per condition (run-1 step-0 minus run-2+ baseline):")
    for c in conditions:
        print(f"  {c:<45s}  {init_times.get(c, 0.0):6.2f} s")


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
