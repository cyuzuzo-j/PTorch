# Experiments: CNN on CIFAR-10

**Thesis:** §5.3 (small CNN on CIFAR-10).
**Directory:** `experiments/neurips/cnn_benchmarks/`.

| Script | Purpose |
|---|---|
| `bench_ptorch.py` | PTorch CNN (`pnn.Conv2D`, `pnn.MaxPool2d`, l2 conv body, configurable `head_norm`) |
| `bench_torch.py`  | Vanilla PyTorch baseline |
| `models.py`       | CNN definitions shared by the scripts |

Everything is set in `resnet8_config.yaml`; plots come from the shared plotter:

```bash
C=experiments/neurips/cnn_benchmarks/resnet8_config.yaml
python -m experiments.neurips.cnn_benchmarks.bench_ptorch --config $C --max-steps 10 --num-runs 1
python -m experiments.neurips.cnn_benchmarks.bench_torch  --config $C --max-steps 10 --num-runs 1
python -m experiments.neurips.plot_curves                 --config $C
```

The alternating gradient/projection and torch-backbone/ptorch-head hybrids
(`bench_hybrid.py`, `bench_ptorch_hybrid_arch.py` and their plots) are not
paper experiments and live in `experiments/cnn_hybrid/`.

## How `pnn.Conv2D` projects

`pnn.Conv2D` (`src/ptorch/nn/modules.py:219-290`) implements convolution by
**unfolding** patches with `ConvPatchProjection` and then projecting the
resulting `(N, H_out, W_out, C·kH·kW)` activation through an *internal*
`pnn.Linear(use_cache=False)`. The bias-fold trick from [[Concepts-Linear-Layer-Projection]]
applies unchanged. The spatial consensus is handled by the fold step inside
`ConvPatchProjection.backward` — see `src/ptorch/core/ops.py:995-1055`.

## How it maps to the thesis

| §5.3 claim | Code |
|---|---|
| CIFAR-10 small-CNN accuracy vs torch baseline | `bench_ptorch.py`, `bench_torch.py`, `plot_curves.py` |

## See also

- [[Reference-NN-Modules]] — `Conv2D`, `MaxPool2d` semantics.
- [[Reference-Ops]] — `ConvPatchProjection`, `MaxPool2DProjection`.
- [[Practical-Architecture-Init]] — App. B.1 warning about mixing layer
  types (relevant to the hybrid-architecture experiment).
