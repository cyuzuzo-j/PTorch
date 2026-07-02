# PTorch

> **Attribution:** A significant portion of this code is based on [AndreasBergmeister/pjax](https://github.com/AndreasBergmeister/pjax).

A PyTorch-based framework for training neural networks via **cyclic projections** instead of backpropagation. Rather than computing gradients, each layer's backward pass finds the nearest point satisfying its local constraint (a projection), and optimizers consume these projection targets as pseudo-gradients.

## How it works

Standard training: `loss.backward()` propagates gradients.  
ptorch: `loss.backward()` propagates *projection targets* — each layer projects its inputs/outputs onto the constraint set defined by its operation, and the optimizer updates parameters by `p ← p - lr * (p - p_proj)`.

## Installation

Requires Python 3.9+.

```bash
# From PyPI
pip install projtorch

# Or from source
git clone https://github.com/cyuzuzo-j/PTorch.git
cd PTorch
pip install .
```

The distribution is named `projtorch`; the import name is `ptorch` (core
dependencies: `torch`, `numpy`).

To run the benchmark/experiment scripts under `experiments/`, install the extra
dependencies too:

```bash
pip install "projtorch[experiments]"   # pandas, tqdm, pyyaml
# add [test] for the pytest suite:    pip install "projtorch[experiments,test]"
```

## Quick start

```python
import ptorch                          # applies projection overrides to torch
import ptorch.nn.modules as pnn
import ptorch.optim_static as poptim
from ptorch.config import config

model = pnn.Linear(784, 10)
criterion = pnn.CrossEntropy()
optimizer = poptim.ProjectionSGD(model.parameters(), lr=1.0)

logits = model(x)
criterion(logits, y_onehot).sum().backward()
optimizer.step()
optimizer.zero_grad()
```

## Modules

| ptorch module | Equivalent |
|---|---|
| `pnn.Linear(in, out, norm='l2'/'linf')` | `nn.Linear` |
| `pnn.ReLU(norm='l2'/'linf')` | `nn.ReLU` |
| `pnn.LeakyReLU()` | `nn.LeakyReLU` |
| `pnn.Conv2D(...)` | `nn.Conv2d` |
| `pnn.MaxPool2d(...)` | `nn.MaxPool2d` |

### Loss functions

| Class | Notes |
|---|---|
| `pnn.CrossEntropy` | Standard cross-entropy via projection |
| `pnn.HardMarginLoss(delta=1.0)` | Hard-margin classifier loss |
| `pnn.ProximalHingeMarginLoss(lambda_val=1.0)` | Soft proximal hinge |

### Optimizers (`ptorch.optim_static`)

All wrap their standard PyTorch counterpart and convert projection targets to pseudo-gradients before the update step.

- `ProjectionSGD` — wraps `torch.optim.SGD` (default `lr=1.0`)
- `ProjectionMuon` — wraps `torch.optim.Muon`
- `ProjectionAdam`, ...

## Examples

### Notebooks

| Notebook | Description |
|---|---|
| `mnist_from_scratch.ipynb` | MNIST classification built manually without ptorch abstractions — best starting point to understand the algorithm |

### MLP benchmark (MNIST / CIFAR-10)

```bash
cd experiments/mlp

# Run ptorch benchmark with default config (linf norm, ProjectionMuon, CrossEntropy)
python bench_ptorch.py --config config.yaml

# Run baseline PyTorch (Adam) for comparison
python bench_torch.py --config config.yaml

# Plot results
python plot_mlp_benchmark.py
```

Edit `config.yaml` to sweep norms (`l2`/`linf`), optimizers, and loss functions.

### CNN benchmark (CIFAR-10)

```bash
cd experiments/cnn_benchmarks

# ptorch CNN
python bench_ptorch.py --config config.yaml

# Baseline
python bench_torch.py --config config.yaml

python plot_results.py
```

### Attention / ViT benchmark (CIFAR-10)

```bash
cd experiments/attention

# ptorch ViT (Projection optimizers)
python bench_ptorch_vit.py --config config.yaml

# Baseline (AdamW)
python bench_torch_vit.py --config config.yaml

python plot_results.py
```

### Non-differentiable activations (MNIST)

Trains MLPs with piecewise-constant activations (Step, GappedStep, QuantizedRelu, Sort) — networks autograd cannot handle.

```bash
cd experiments/non_differentiable
python quantized_relu.py --config config.yaml
python plot_results.py
```

### Deep network analysis

Theoretical/empirical studies on deep linear MLPs.

```bash
# Local non-expansiveness of the (forward, backward target) projection pair across depths
python experiments/deep/local_nonexpansiveness_deep.py

# Vanishing target signal as it backpropagates through depth
python experiments/deep/vanishing_target.py
```

Both scripts use hardcoded constants at the top of the file (edit them to scale runs up/down).

### Quick smoke test

Every config-driven benchmark accepts `--max-steps N --num-runs M` to override the YAML for a fast end-to-end check:

```bash
python experiments/mlp/bench_ptorch.py --max-steps 10 --num-runs 1
python experiments/cnn_benchmarks/bench_ptorch.py --max-steps 10 --num-runs 1
python experiments/attention/bench_ptorch_vit.py --max-steps 5 --num-runs 1
```

