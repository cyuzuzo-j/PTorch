# PTorch

> **Attribution:** A significant portion of this code is based on or directly copied from [AndreasBergmeister/pjax](https://github.com/AndreasBergmeister/pjax).

A PyTorch-based framework for training neural networks via **cyclic projections** instead of backpropagation. Rather than computing gradients, each layer's backward pass finds the nearest point satisfying its local constraint (a projection), and optimizers consume these projection targets as pseudo-gradients.

## How it works

Standard training: `loss.backward()` propagates gradients.  
ptorch: `loss.backward()` propagates *projection targets* — each layer projects its inputs/outputs onto the constraint set defined by its operation, and the optimizer updates parameters by `p ← p - lr * (p - p_proj)`.

Importing `ptorch` automatically overrides `torch.sum`, `torch.add`, `torch.mul`, and `torch.square` with projection-aware versions.

## Installation

```bash
# Clone the repo
git clone <repo-url>
cd learning_with_projections

# Install dependencies (Python 3.10+)
pip install torch numpy pandas tqdm pyyaml
```

The `frameworks/` directory is used directly from source — no `pip install` needed. Scripts add it to `sys.path` automatically.

## Quick start

```python
import sys
sys.path.insert(0, "frameworks")

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
| `xor_from_scratch.ipynb` | Cyclic projections on XOR from scratch in JAX — best starting point to understand the algorithm |
| `mnist_from_scratch.ipynb` | MNIST classification built manually without ptorch abstractions |

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

