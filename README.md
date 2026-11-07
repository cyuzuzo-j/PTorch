# PTorch

> **Attribution:** A significant portion of this code is based on [AndreasBergmeister/pjax](https://github.com/AndreasBergmeister/pjax), which is also included as the PJAX baseline (`third_party/pjax`).

A PyTorch-based framework for training neural networks via **cyclic projections** instead of backpropagation. Rather than computing gradients, each layer's backward pass finds the nearest point satisfying its local constraint (a projection), and optimizers consume these projection targets as pseudo-gradients.

This repository contains the `ptorch` library and the code to reproduce every figure and table of the NeurIPS paper (`experiments/neurips/`).

## How it works

Standard training: `loss.backward()` propagates gradients.
ptorch: `loss.backward()` propagates *projection targets* — each layer projects its inputs/outputs onto the constraint set defined by its operation, and the optimizer updates parameters by `p ← p - lr * (p - p_proj)`.

The linear projection is solved in fused form: the per-sample, per-output projections are averaged in closed form with ordinary matrix products, so the `batch × out_features × in_features` tensor of individual projections is never formed.

## Installation

Requires Python 3.10+ and PyTorch 2.14+ (`torch.optim.Muon` needs 2.9+, and Inductor in 2.9–2.11 miscompiles one of the GPU kernels).

```bash
git clone --recurse-submodules https://github.com/cyuzuzo-j/PTorch.git
cd PTorch
pip install .
```

The import name is `ptorch` (core dependencies: `torch`, `numpy`).

To run the experiments, install the extra dependencies too, or the full pinned environment:

```bash
pip install ".[experiments]"     # torchvision, pandas, tqdm, pyyaml, matplotlib, seaborn, ...
pip install ".[experiments,jax]" # + JAX, for the PJAX baseline
pip install ".[dev]"             # pytest, ruff
pip install -r requirements.txt  # everything, ptorch installed editable
```

## Quick start

```python
import torch.nn.functional as F
import ptorch                          # applies projection overrides to torch
import ptorch.nn.modules as pnn
import ptorch.optim_static as poptim

model = pnn.Linear(784, 10)            # norm='l2' (default) or 'linf'
criterion = pnn.CrossEntropy()         # proximal cross-entropy
optimizer = poptim.ProjectionSGD(model.parameters(), lr=1.0)

logits = model(x)
criterion(logits, F.one_hot(y, 10).float()).sum().backward()
optimizer.step()
optimizer.zero_grad()
```

Global settings (projection strength, proximal CE λ, ℓ∞ iterations, ...) live in `ptorch.config.config`; change them with `config.update("name", value)` or the matching `PTORCH_*` environment variable.

## Modules

| ptorch module | Equivalent |
|---|---|
| `pnn.Linear(in, out, norm='l2'/'linf')` | `nn.Linear` |
| `pnn.ReLU(norm='l2'/'linf')` | `nn.ReLU` |
| `pnn.LeakyReLU()` | `nn.LeakyReLU` |
| `pnn.Softmax()` | `nn.Softmax` |
| `pnn.Conv2D(...)` | `nn.Conv2d` |
| `pnn.MaxPool2d(...)` | `nn.MaxPool2d` |
| `pnn.Step()`, `pnn.GappedStep()`, `pnn.QuantizedRelu()` | non-differentiable activations (no autograd equivalent) |

### Loss functions

| Class | Notes |
|---|---|
| `pnn.CrossEntropy` | Proximal cross-entropy (λ = 10, one inner step by default) |
| `pnn.HardMarginLoss(delta=1.0)` | Hard-margin target assignment |
| `pnn.ProximalHingeMarginLoss(lambda_val=1.0)` | Soft proximal hinge |

### Optimizers (`ptorch.optim_static`)

All wrap their standard PyTorch counterpart and convert projection targets to pseudo-gradients before the update step.

- `ProjectionSGD` — wraps `torch.optim.SGD` (default `lr=1.0`)
- `ProjectionMuon` — wraps `torch.optim.Muon`
- `ProjectionAdam`, `ProjectionAdagrad`, `ProjectionAdadelta`

## Reproducing the paper

Every experiment is one YAML file; the benches, plots and tables all take it with `--config`. Run from the repository root:

```bash
C=experiments/neurips/mlp/config.yaml                                # shallow MLP, MNIST + CIFAR-10
python -m experiments.neurips.mlp.bench_ptorch --config $C --fresh   # PTorch variants
python -m experiments.neurips.mlp.bench_torch  --config $C --fresh   # Torch (Adam) baseline
python -m experiments.neurips.mlp.bench_pjax   --config $C --fresh   # PJAX baseline (third_party/pjax on PYTHONPATH)
python -m experiments.neurips.plot_curves      --config $C           # figures
python -m experiments.neurips.make_tables      --config $C           # LaTeX test tables
```

Every bench accepts `--max-steps N --num-runs M` for a quick smoke test, e.g. `--max-steps 10 --num-runs 1`.

| Paper item | Config |
|---|---|
| Shallow MLP, MNIST + CIFAR-10 | `experiments/neurips/mlp/config.yaml` |
| 8-layer MLP | `experiments/neurips/mlp/deep8_config.yaml` |
| Vanishing target signal | `experiments/neurips/deep/vanishing_target_config.yaml` |
| ResNet-8 on CIFAR-10 | `experiments/neurips/cnn_benchmarks/resnet8_config.yaml` |
| CNN memory vs PJAX | `experiments/neurips/cnn_benchmarks/memory_config.yaml` |
| Weight initialization | `experiments/neurips/deep/effect_of_init_config.yaml` |
| Effect of the loss | `experiments/neurips/mlp/effect_of_loss_config.yaml` |
| Optimizer comparison | `experiments/neurips/mlp/optimizer_comparison_config.yaml` |
| Memory vs input dimension / batch size | `experiments/neurips/mlp/memory_config.yaml`, `memory_batch_config.yaml` |
| ℓ₂ vs ℓ∞ | `experiments/neurips/mlp/norm_comparison_config.yaml` |
| Ablation | `experiments/neurips/mlp/ablation_config.yaml`, `ablation_memory_config.yaml` |
| Relaxation location on XOR | `experiments/neurips/xor/xor_from_scratch.ipynb` |

See [`experiments/neurips/README.md`](experiments/neurips/README.md) for the data protocol (train/val/test splits), the full paper-to-config mapping and the variant keys.

### Notebook

| Notebook | Description |
|---|---|
| `experiments/neurips/xor/xor_from_scratch.ipynb` | XOR classifier written from scratch in JAX (`xor/tools/projections.py`), without the ptorch abstractions — the best starting point to understand the algorithm |

## Tests

```bash
pip install ".[dev]"
pytest
```

## License

MIT, see [LICENSE](LICENSE).
