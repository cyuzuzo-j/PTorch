"""Run experiments/*/plot_*.py to refresh docs_src/images/.

Each plot script is expected to honour the `figures_dir:` key in its YAML
config (the MLP/CNN/attention scripts do; the deep-diagnostics scripts use
hardcoded constants instead). Where a script can't be redirected we just
copy whatever it already produced into the wiki image tree.

This script is conservative: it skips experiments whose `results/` directory
is empty (no checkpoints to plot from) and prints a TODO instead of running
a benchmark — those are expensive and are not the wiki's job to launch.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


PLOT_TARGETS = [
    # (module path, expected source dir, output subfolder under docs_src/images)
    ("experiments.mlp.plot_mlp_benchmark",     "experiments/mlp/results",            "mlp"),
    ("experiments.mlp.plot_optimizer_comparison", "experiments/mlp/results",          "mlp"),
    ("experiments.mlp.plot_norm_comparison",   "experiments/mlp/results",            "mlp"),
    ("experiments.mlp.plot_effect_of_loss",    "experiments/mlp/results",            "mlp"),
    ("experiments.mlp.plot_deep_mlp",          "experiments/mlp/results",            "mlp"),
    ("experiments.mlp.plot_memory",            "experiments/mlp/results",            "mlp"),
    ("experiments.cnn_benchmarks.plot_results", "experiments/cnn_benchmarks/results", "cnn"),
    ("experiments.cnn_benchmarks.plot_hybrid",  "experiments/cnn_benchmarks/results", "cnn"),
    ("experiments.cnn_benchmarks.plot_hybrid_arch", "experiments/cnn_benchmarks/results", "cnn"),
    ("experiments.attention.plot_results",     "experiments/attention/results",      "attention"),
    ("experiments.non_differentiable.plot_results", "experiments/non_differentiable/results", "non_differentiable"),
]

DEEP_IMAGE_SOURCES = [
    ("experiments/deep/images", "deep"),
]


def has_results(results_dir: Path) -> bool:
    if not results_dir.is_dir():
        return False
    return any(p.suffix in {".csv", ".json", ".jsonl"} for p in results_dir.rglob("*"))


def run_plot(module: str, log_prefix: str) -> bool:
    print(f"== {log_prefix} running {module}")
    rc = subprocess.call(["python", "-m", module], cwd=REPO_ROOT)
    if rc != 0:
        print(f"   FAILED ({rc}); skipping")
    return rc == 0


def harvest_images(produced_root: Path, out_dir: Path) -> int:
    """Copy any newly produced PNGs/PDFs from produced_root into out_dir."""
    if not produced_root.is_dir():
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for ext in ("*.png", "*.pdf"):
        for f in produced_root.rglob(ext):
            shutil.copy2(f, out_dir / f.name)
            count += 1
    return count


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "docs_src" / "images",
        help="root output directory (default: docs_src/images/)",
    )
    ap.add_argument(
        "--from-images-only",
        action="store_true",
        help="skip plot script execution; just harvest whatever images already exist",
    )
    args = ap.parse_args()

    out_root: Path = args.out.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    todos: list[str] = []

    for module, results_rel, out_sub in PLOT_TARGETS:
        results_dir = REPO_ROOT / results_rel
        if not has_results(results_dir):
            todos.append(
                f"  TODO: {results_rel} is empty — run the benchmark to populate, "
                f"then re-run this script ({module})"
            )
            continue
        if not args.from_images_only:
            run_plot(module, "[plot]")
        # Heuristic: most plot scripts dump under images/<sub>/ at repo root,
        # but a few drop straight into experiments/*/plots or images/.
        candidates = [
            REPO_ROOT / "images" / out_sub,
            REPO_ROOT / "images",
            REPO_ROOT / Path(results_rel).parent / "plots",
            REPO_ROOT / Path(results_rel).parent / "images",
        ]
        moved = 0
        for c in candidates:
            moved += harvest_images(c, out_root / out_sub)
        if moved == 0:
            todos.append(
                f"  TODO: {module} produced no images we could find in "
                f"{[str(c.relative_to(REPO_ROOT)) for c in candidates]}"
            )

    for src_rel, out_sub in DEEP_IMAGE_SOURCES:
        src = REPO_ROOT / src_rel
        moved = harvest_images(src, out_root / out_sub)
        if moved == 0:
            todos.append(
                f"  TODO: no images in {src_rel} — run the deep-diagnostics scripts "
                f"(see Experiments-Deep-Diagnostics)"
            )

    print()
    print(f"figures landed under {out_root}")
    if todos:
        print("Pending items:")
        for t in todos:
            print(t)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
