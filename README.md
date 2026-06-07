# PTorch

> **Attribution:** A significant portion of this code is based on [AndreasBergmeister/pjax](https://github.com/AndreasBergmeister/pjax).

Full documentation lives on the [PTorch Wiki](https://github.com/cyuzuzo-j/PTorch/wiki) — code-first reference with thesis cross-links. Wiki source is in `docs_src/`; sync via `./tools/wiki/publish.sh`.

A PyTorch-based framework for training neural networks via **cyclic projections** instead of backpropagation. Rather than computing gradients, each layer's backward pass finds the nearest point satisfying its local constraint (a projection), and optimizers consume these projection targets as pseudo-gradients.

## How it works

Standard training: `loss.backward()` propagates gradients.  
ptorch: `loss.backward()` propagates *projection targets* — each layer projects its inputs/outputs onto the constraint set defined by its operation, and the optimizer updates parameters by `p ← p - lr * (p - p_proj)`.

## Installation

Requires Python 3.10+.

```bash
# Editable install (core + experiments + dev tooling)
pip install -e ".[experiments,dev]"

# pjax is not on PyPI — install from GitHub
pip install git+https://github.com/AndreasBergmeister/pjax.git
```

For a minimal install without the benchmark stack: `pip install -e .`

## Quick start

```python
import torch
import torch.nn.functional as F
import ptorch                          # applies projection overrides to torch
import ptorch.nn.modules as pnn
import ptorch.optim_static as poptim
from ptorch.config import config

# Dummy MNIST-shaped batch
x = torch.randn(8, 784)
y = torch.randint(0, 10, (8,))
y_onehot = F.one_hot(y, num_classes=10).float()

model = pnn.Linear(784, 10)
criterion = pnn.CrossEntropy()
optimizer = poptim.ProjectionSGD(model.parameters(), lr=1.0)

logits = model(x)
criterion(logits, y_onehot).sum().backward()
optimizer.step()
optimizer.zero_grad()
```

> **First-run latency.** The first import takes ~30–60 s because several hot
> projection ops (`zeropower_via_polarexpress`, `process_activation_target`,
> `matmul_proj_linf`, `matmul_proj`) are wrapped in `@torch.compile()`. After
> the first trace they're fast for the rest of the session. If you're
> experimenting locally and don't need full throughput, comment out the
> `@torch.compile(...)` decorators on those functions in
> `src/ptorch/core/ops.py` and `src/ptorch/optim_static.py` —
> everything still runs in
> eager mode, just slower.

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

All commands below are run from the **repo root** as modules (`python -m ...`) so that `from experiments.shared.data import ...` resolves cleanly.

### MLP benchmark (MNIST / CIFAR-10)

```bash
# Run ptorch benchmark with default config (linf norm, ProjectionMuon, CrossEntropy)
python -m experiments.mlp.bench_ptorch --config experiments/mlp/config.yaml

# Run baseline PyTorch (Adam) for comparison
python -m experiments.mlp.bench_torch --config experiments/mlp/config.yaml

# Plot results
python -m experiments.mlp.plot_mlp_benchmark
```

Edit `experiments/mlp/config.yaml` to sweep norms (`l2`/`linf`), optimizers, and loss functions.

### CNN benchmark (CIFAR-10)

```bash
python -m experiments.cnn_benchmarks.bench_ptorch --config experiments/cnn_benchmarks/config.yaml
python -m experiments.cnn_benchmarks.bench_torch  --config experiments/cnn_benchmarks/config.yaml
python -m experiments.cnn_benchmarks.plot_results
```

### Attention / ViT benchmark (CIFAR-10)

```bash
python -m experiments.attention.bench_ptorch_vit --config experiments/attention/config.yaml
python -m experiments.attention.bench_torch_vit  --config experiments/attention/config.yaml
python -m experiments.attention.plot_results
```

### Non-differentiable activations (MNIST)

Trains MLPs with piecewise-constant activations (Step, GappedStep, QuantizedRelu) — networks autograd cannot handle.

```bash
python -m experiments.non_differentiable.quantized_relu --config experiments/non_differentiable/config.yaml
python -m experiments.non_differentiable.plot_results
```

### Deep network analysis

Theoretical/empirical studies on deep linear MLPs.

```bash
# Local non-expansiveness of the (forward, backward target) projection pair across depths
python -m experiments.deep.local_nonexpansiveness_deep

# Vanishing target signal as it backpropagates through depth
python -m experiments.deep.vanishing_target
```

Both scripts use hardcoded constants at the top of the file (edit them to scale runs up/down).

### Quick smoke test

Every config-driven benchmark accepts `--max-steps N --num-runs M` to override the YAML for a fast end-to-end check:

```bash
python -m experiments.mlp.bench_ptorch --max-steps 10 --num-runs 1
python -m experiments.cnn_benchmarks.bench_ptorch --max-steps 10 --num-runs 1
python -m experiments.attention.bench_ptorch_vit --max-steps 5 --num-runs 1
```

