"""Test-set tables (mean ± std over seeds) for any NeurIPS experiment.

    python -m experiments.neurips.make_tables --config experiments/neurips/mlp/ablation_config.yaml

Reads the single `split == test` row each run writes (the best-validation
checkpoint evaluated once on the official test set) and prints one LaTeX
tabular per task. The YAML's optional `table_by:` adds a grouping column
(e.g. label_noise). Columns a CSV does not carry are left out. Peak memory
is not reported here: reduce-overhead's cudagraph pools make the training
runs' allocator peak unreliable, so memory comes from bench_memory only.
"""
import argparse
import os

from experiments.neurips import common

# Optional columns, in display order: (csv column, header, decimals)
EXTRA = [("ms_per_step", "ms/step", 2),
         ("compile_s", "Compile (s)", 1), ("wall_time_s", "Train time (s)", 1)]


def pm(series, scale=1.0, decimals=2):
    s = series.dropna() * scale
    if s.empty:
        return "---"
    if len(s) == 1:
        return f"${s.iloc[0]:.{decimals}f}$"
    return f"${s.mean():.{decimals}f} \\pm {s.std(ddof=1):.{decimals}f}$"


def table(df, task, by=None):
    sub = df[df["task"] == task]
    extras = [e for e in EXTRA if e[0] in sub.columns and sub[e[0]].notna().any()]
    keys = ["Label"] + ([by] if by else [])
    head = (["Method"] + ([by.replace("_", " ")] if by else []) + ["Test acc (\\%)"]
            + [h for _, h, _ in extras] + ["runs"])
    lines = [f"% {task}", "\\begin{tabular}{l" + "c" * (len(head) - 1) + "}", "\\hline",
             " & ".join(f"\\textbf{{{h}}}" for h in head) + " \\\\", "\\hline"]
    for key, g in sub.groupby(keys, sort=True):
        cells = [str(k) for k in key] + [pm(g["acc"], 100.0)]
        cells += [pm(g[c], 1.0, d) for c, _, d in extras] + [str(g["seed"].nunique())]
        lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(lines + ["\\hline", "\\end{tabular}"])


def _num(x, decimals=1):
    return "---" if x != x else f"{x:,.{decimals}f}"  # x != x: NaN


def ablation_table(cfg, df):
    """table:ablation, one row per `labels:` entry in YAML order. Test acc is a
    fraction; ms/step comes from the compiled runs, ms/step (eager) from the
    `<name>_eager` runs, Test acc and Wall from the compiled runs (eager if a row has none), Peak MB
    from the eager short runs in `memory_results_dir` (allocator peak of the training
    steps, data excluded; rows not run there use their own runs' peak, e.g. PJAX)."""
    mem = common.load_runs({**cfg, "results_dir": cfg["memory_results_dir"]}, split="test")
    lines = ["\\begin{tabular}{l c c c c c}", "\\hline",
             "Method & Test acc. & ms/step & ms/step (eager) & Peak MB & Wall (s) \\\\", "\\hline"]
    for name, label in cfg["labels"].items():
        runs, eager = df[df["variant"] == name], df[df["variant"] == f"{name}_eager"]
        if runs.empty and eager.empty:
            continue
        main = runs if len(runs) else eager
        m = mem[mem["variant"].isin([name, f"{name}_eager"])]["peak_mb"]
        if m.empty and "peak_mb" in runs:
            m = runs["peak_mb"]
        lines.append(" & ".join([
            label, pm(main["acc"], 1.0, 4), pm(runs["ms_per_step"]), pm(eager["ms_per_step"]),
            _num(m.mean() if len(m) else float("nan")), _num(main["wall_time_s"].mean())]) + " \\\\")
    return "\n".join(lines + ["\\hline", "\\end{tabular}"])


def compile_times(cfg, df):
    """Compile time of each table:ablation row over its seeds (the text around the table):
    saved to results_dir/compile_times.csv and printed as LaTeX comments."""
    runs = df[df["variant"].str.removesuffix("_eager").isin(cfg["labels"])].dropna(subset=["compile_s"])
    stats = runs.groupby("variant")["compile_s"].agg(["mean", "std", "min", "max", "count"])
    stats.to_csv(os.path.join(cfg["results_dir"], "compile_times.csv"))
    return "\n".join(f"% compile {name}: {s['mean']:.1f} +- {s['std']:.1f} s (min {s['min']:.1f}, max {s['max']:.1f})"
                     for name, s in stats.iterrows())


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", required=True, help="experiment YAML")
    cfg = common.load_config(p.parse_args().config)
    df = common.load_runs(cfg, split="test")
    if df.empty:
        raise SystemExit(f"no test rows in {cfg['results_dir']}")
    if cfg.get("table") == "ablation":
        print(ablation_table(cfg, df))
        print(compile_times(cfg, df))
        return
    df.loc[df["Label"] == df["variant"], "Label"] = df["variant"].str.replace("_", r"\_")  # unlabeled: LaTeX-safe
    for task in sorted(df["task"].unique()):
        print(table(df, task, cfg.get("table_by")))
        print()


if __name__ == "__main__":
    main()
