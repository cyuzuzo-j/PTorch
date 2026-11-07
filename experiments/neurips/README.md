# NeurIPS paper experiments

Everything needed to reproduce the figures and tables of the PTorch NeurIPS
submission. Research probes and side projects live in the other `experiments/`
folders.

## How it works

**One YAML per experiment.** It holds the tasks, seeds, step budget, every
variant, the optimizers, label noise, time budget, output folders, display
labels and the plots. Every script takes it with `--config` and nothing else
changes the experiment. Paths in a YAML are relative to the YAML.

Run from the repository root:

```bash
C=experiments/neurips/mlp/config.yaml
python -m experiments.neurips.mlp.bench_ptorch --config $C --fresh   # PTorch variants (ptorch:)
python -m experiments.neurips.mlp.bench_torch  --config $C --fresh   # Torch baseline  (torch:)
python -m experiments.neurips.mlp.bench_pjax   --config $C --fresh   # PJAX baseline   (pjax:, needs third_party/pjax on PYTHONPATH)
python -m experiments.neurips.plot_curves      --config $C           # figures         (plots:)
python -m experiments.neurips.make_tables      --config $C           # LaTeX test tables
```

The benches also take `--tasks`/`--variants` (run a subset) and
`--max-steps`/`--num-runs` (smoke tests), e.g. `--max-steps 10 --num-runs 1`.
`--fresh` deletes old CSVs before writing.

**Data protocol** (`common.py`, checked by `tests/test_data_splits.py`):

| Split | Source | Used for |
|---|---|---|
| train | seeded 90% of the official training set | training |
| val | the other 10% | the curves, early stopping, picking the best checkpoint |
| test | the official test set | one evaluation per run, on the best-val checkpoint |

The split is seeded by the run seed (`random_seed + r`), so every framework
sees the same split and batch order for a given run. Label noise only changes
training labels. Every CSV row has `split` = `val` or `test`. Plots read only
`val` rows and tables read only `test` rows.

## Paper → YAML

Settings the paper does not state follow Experiment 1 and `tab:shared-protocol`.

| Paper item | YAML | Scripts |
|---|---|---|
| Shallow MLP, MNIST + CIFAR-10 (`fig:shallow-mlp`, `tab:test-results`) | `mlp/config.yaml` | ptorch, torch, pjax benches; plot_curves; make_tables |
| Vanishing targets (a): identity recovery (`fig:depth-barrier`) | `deep/vanishing_target_config.yaml` | `deep/vanishing_target.py` |
| Vanishing targets (b): 8-layer MLP (`fig:deep-mlp`, `tab:test-results`) | `mlp/deep8_config.yaml` | ptorch, torch benches; plot_curves; make_tables |
| ResNet-8 on CIFAR-10 (`fig:resnet-8`) | `cnn_benchmarks/resnet8_config.yaml` | `cnn_benchmarks/bench_{ptorch,torch}.py`; plot_curves |
| CNN memory, PJAX / PTorch (`fig:mem-cnn`) | `cnn_benchmarks/memory_config.yaml` | `cnn_benchmarks/bench_memory.py`, `cnn_benchmarks/plot_memory.py` |
| Weight initialization (`fig:effect-of-init`) | `deep/effect_of_init_config.yaml` | ptorch bench; plot_curves |
| Proximal vs hard loss (`fig:effect-of-loss`) | `mlp/effect_of_loss_config.yaml` | ptorch bench; plot_curves |
| Optimizer comparison (`fig:opt`) | `mlp/optimizer_comparison_config.yaml` | ptorch bench; plot_curves |
| Memory vs input dimension / batch size (`fig:memory`) | `mlp/memory_config.yaml`, `mlp/memory_batch_config.yaml` | `mlp/bench_memory.py`, `mlp/plot_memory.py` |
| ℓ₂ vs ℓ∞ wall-clock (`fig:norm-comparison`) | `mlp/norm_comparison_config.yaml` | ptorch bench; plot_curves |
| Backward passes K, equal compute / equal time (`tab:backward-passes`) | `mlp/backward_passes_config.yaml`, `mlp/backward_passes_equal_time_config.yaml` | ptorch bench; make_tables |
| Unified ablation (`table:ablation`) | `mlp/ablation_config.yaml`, `mlp/ablation_memory_config.yaml` (Peak MB) | ptorch, torch benches; make_tables |
| Relaxation location on XOR (`fig:relaxation-location`) | `mlp/xor_relaxation_config.yaml` | `mlp/xor_relaxation.py` |
| Label-noise robustness | `mlp/label_noise_config.yaml` | ptorch, torch benches; make_tables |

## Files

```
common.py          config loading, data splits, training loop, CSV I/O
plot_curves.py     validation curves from a YAML's plots: list
make_tables.py     LaTeX test-set tables (mean ± std over seeds)
style.py           figure colours (the paper's Figure 1 palette), shared by every plot
mlp/               MLP benches (ptorch, torch, pjax), unfused_linear.py (ablation R1), memory bench
cnn_benchmarks/    CNN models and benches
deep/              vanishing-target probe
```

PTorch variant keys and their defaults are listed at the top of
`mlp/bench_ptorch.py` and `cnn_benchmarks/bench_ptorch.py` (`VARIANT_DEFAULTS`).
A misspelled key is an error, not silently ignored.
