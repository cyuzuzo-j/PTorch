# Experiments: shallow MLP

**Thesis:** §5.1 (Shallow MLP: MNIST and CIFAR-10).
**Directory:** `experiments/neurips/mlp/`.

The shallow-MLP benchmark is the headline result: PTorch's projection-based
training matches gradient-trained baselines (Adam) on single-hidden-layer
networks for MNIST and CIFAR-10. Every panel in §5.1 / §5.2 of the thesis is
driven by a YAML config + a plot script in this folder.

## How the experiments are run

Each experiment is one YAML; every script takes it with `--config` (see
`experiments/neurips/README.md` for the full paper → YAML map and the
train/val/test protocol).

| Script | Purpose |
|---|---|
| `mlp/bench_ptorch.py` | PTorch variants listed under `ptorch:` in the YAML |
| `mlp/bench_torch.py`  | PyTorch baseline (`torch:` block) |
| `mlp/bench_pjax.py`   | PJAX baseline (`pjax:` block; needs the `third_party/pjax` submodule) |
| `mlp/bench_memory.py`, `mlp/plot_memory.py` | Peak memory vs input dim (`memory_config.yaml`) |
| `plot_curves.py`      | Validation curves from the YAML's `plots:` list |
| `make_tables.py`      | LaTeX test-set tables |

| Config | Paper item |
|---|---|
| `config.yaml`                       | Shallow MLP, MNIST_1L + CIFAR10_1L |
| `deep8_config.yaml`                 | 8-layer MLP, floor(λ) on/off |
| `ablation_config.yaml`              | Unified ablation + hybrid MLP table |
| `norm_comparison_config.yaml`       | `l2` vs `linf` over wall-clock time |
| `optimizer_comparison_config.yaml`  | SGD vs Adam vs Adadelta vs Muon on the projection residual |
| `effect_of_loss_config.yaml`        | `CrossEntropy` vs `HardMargin` vs `ProximalHingeMargin` |
| `backward_passes*_config.yaml`      | K backward passes per batch, equal compute / equal time |
| `label_noise_config.yaml`           | Label-noise robustness |
| `memory_config.yaml`                | Peak memory vs PJAX |

Results land in the `results_dir` of each YAML (under
`experiments/neurips/mlp/results/`); figures go to its `figures_dir`.

## Smoke test

```bash
C=experiments/neurips/mlp/config.yaml
python -m experiments.neurips.mlp.bench_ptorch --config $C --max-steps 10 --num-runs 1
python -m experiments.neurips.mlp.bench_torch  --config $C --max-steps 10 --num-runs 1
python -m experiments.neurips.plot_curves      --config $C
```

Drop `--max-steps`/`--num-runs` for the paper numbers. Regenerate the wiki
figures with `python tools/wiki/build_figures.py` once results exist.

## See also

- [[Concepts-Linear-Layer-Projection]] — `norm` choice behind the
  `norm_comparison` sweep.
- [[Reference-Optimizers]] — optimizer wrappers tested by
  `optimizer_comparison`.
- [[Experiments-Deep-Diagnostics]] — the `deep8` config's results feed
  the vanishing-targets analysis.
