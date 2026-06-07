# Experiments: CNN on CIFAR-10

**Thesis:** §5.3 (small CNN on CIFAR-10), App. A.1–A.2 (hybrid architecture
+ hybrid optimisation).
**Directory:** `experiments/cnn_benchmarks/`.

## Bench scripts

| Script | Purpose |
|---|---|
| `bench_ptorch.py`             | PTorch CNN (uses `pnn.Conv2D`, `pnn.MaxPool2d`, `pnn.ReLU`) |
| `bench_torch.py`              | Vanilla PyTorch baseline |
| `bench_hybrid.py`             | **Hybrid optimisation** — some params updated by projection, others by gradient (`config.use_hybrid=True`; see `ProjectionMuonV2`) |
| `bench_ptorch_hybrid_arch.py` | **Hybrid architecture** — mixes `pnn.*` and `nn.*` layers in the same model. Backs App. A.1 |
| `models.py`                   | CNN definitions shared by the scripts |

All take `--config experiments/cnn_benchmarks/config.yaml` and the standard
`--max-steps`/`--num-runs` overrides.

## Plot scripts and outputs

| Script | Output (under `images/cnn/` by default) |
|---|---|
| `plot_results.py`     | Headline PTorch vs torch accuracy curves |
| `plot_hybrid.py`      | App. A.2 hybrid-optimisation panel |
| `plot_hybrid_arch.py` | App. A.1 hybrid-architecture panel |

## Smoke test

```bash
python -m experiments.cnn_benchmarks.bench_ptorch --max-steps 10 --num-runs 1
python -m experiments.cnn_benchmarks.bench_torch  --max-steps 10 --num-runs 1
python -m experiments.cnn_benchmarks.plot_results
```

## How `pnn.Conv2D` projects

`pnn.Conv2D` (`src/ptorch/nn/modules.py:219-290`) implements convolution by
**unfolding** patches with `ConvPatchProjection` and then projecting the
resulting `(N, H_out, W_out, C·kH·kW)` activation through an *internal*
`pnn.Linear(use_cache=False)`. The bias-fold trick from [[Concepts-Linear-Layer-Projection]]
applies unchanged. The spatial consensus is handled by the fold step inside
`ConvPatchProjection.backward` — see `src/ptorch/core/ops.py:995-1055`.

## How it maps to the thesis

| §5.3 / App. claim | Code |
|---|---|
| CIFAR-10 small-CNN accuracy ≈ torch baseline | `bench_ptorch.py` + `plot_results.py` |
| Hybrid architecture (some layers `pnn.*`, some `nn.*`) | `bench_ptorch_hybrid_arch.py` + `plot_hybrid_arch.py` |
| Hybrid optimisation (`_is_target` flag) | `bench_hybrid.py` + `plot_hybrid.py` |

## See also

- [[Reference-NN-Modules]] — `Conv2D`, `MaxPool2d` semantics.
- [[Reference-Ops]] — `ConvPatchProjection`, `MaxPool2DProjection`.
- [[Practical-Architecture-Init]] — App. B.1 warning about mixing layer
  types (relevant to the hybrid-architecture experiment).
