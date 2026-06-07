# Experiments: shallow MLP

**Thesis:** §5.1 (Shallow MLP: MNIST and CIFAR-10).
**Directory:** `experiments/mlp/`.

The shallow-MLP benchmark is the headline result: PTorch's projection-based
training matches gradient-trained baselines (Adam) on single-hidden-layer
networks for MNIST and CIFAR-10. Every panel in §5.1 / §5.2 of the thesis is
driven by a YAML config + a plot script in this folder.

## Bench scripts

| Script | Purpose |
|---|---|
| `bench_ptorch.py` | PTorch run (`pnn.Linear` + `ProjectionMuon`/`CrossEntropy` by default) |
| `bench_torch.py`  | Vanilla PyTorch baseline (Adam + `nn.CrossEntropyLoss`) |
| `bench_pjax.py`   | PJAX baseline (uses `pip install git+https://github.com/AndreasBergmeister/pjax.git`) |
| `bench_memory.py` | Peak-RSS comparison vs. PJAX — backs the memory claim in §3.1.1 |

All accept `--config <yaml> --max-steps N --num-runs M`. The configs that
matter:

| Config | Sweep |
|---|---|
| `config.yaml`                       | Default headline benchmark (MNIST_1L, CIFAR10_1L; linf; Muon) |
| `norm_comparison_config.yaml`       | `l2` vs `linf` projection geometry |
| `optimizer_comparison_config.yaml`  | SGD vs Adam vs Adagrad vs Muon, all wrapped as projection optimizers |
| `effect_of_loss_config.yaml`        | `CrossEntropy` vs `HardMargin` vs `ProximalHingeMargin` |
| `deep_mlp_config.yaml`              | Depth sweep (feeds into Deep diagnostics) |

Results land in `experiments/mlp/results/` as CSVs; plot scripts read from
there.

## Plot scripts and outputs

| Plot script | Figures saved to |
|---|---|
| `plot_mlp_benchmark.py`     | `images/mlp/` (set by `figures_dir:` in YAML) |
| `plot_optimizer_comparison.py` | `images/mlp/optimizer_comparison.{pdf,png}` |
| `plot_norm_comparison.py`   | `images/mlp/norm_comparison.{pdf,png}` |
| `plot_effect_of_loss.py`    | `images/mlp/effect_of_loss.{pdf,png}` |
| `plot_deep_mlp.py`          | `images/mlp/deep_mlp.{pdf,png}` |
| `plot_memory.py`            | `images/mlp/bench_memory.{pdf,png}` |

The wiki's `images/` folder pulls these via `tools/wiki/build_figures.py`.

## Smoke test

```bash
python -m experiments.mlp.bench_ptorch --max-steps 10 --num-runs 1
python -m experiments.mlp.bench_torch  --max-steps 10 --num-runs 1
python -m experiments.mlp.plot_mlp_benchmark
```

For the full headline numbers: drop the `--max-steps` flag (the YAML uses
`5_000`).

## Headline figure

Regenerate with `python tools/wiki/build_figures.py` once
`experiments/mlp/results/` is populated. The output lands in
`docs_src/images/mlp/` and can be embedded here as
`![](images/mlp/<filename>.png)` once the exact filename is known.

## How it maps to the thesis

| §5.1 figure | YAML | Plot |
|---|---|---|
| Headline accuracy curves | `config.yaml` | `plot_mlp_benchmark.py` |
| Optimizer ablation | `optimizer_comparison_config.yaml` | `plot_optimizer_comparison.py` |
| Norm ablation | `norm_comparison_config.yaml` | `plot_norm_comparison.py` |
| Loss ablation | `effect_of_loss_config.yaml` | `plot_effect_of_loss.py` |
| Peak memory vs PJAX | `bench_memory.py` (script-level config) | `plot_memory.py` |

## See also

- [[Concepts-Linear-Layer-Projection]] — `norm` choice behind the
  `norm_comparison` sweep.
- [[Reference-Optimizers]] — optimizer wrappers tested by
  `optimizer_comparison`.
- [[Experiments-Deep-Diagnostics]] — the `deep_mlp` config's results feed
  the vanishing-targets analysis.
