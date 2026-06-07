# Getting Started

## Install

Python 3.10+. Editable install with the experiments stack:

```bash
pip install -e ".[experiments,dev]"
pip install git+https://github.com/AndreasBergmeister/pjax.git   # not on PyPI
```

Minimal install (no benchmarks): `pip install -e .`

## First-run latency

The first `import ptorch` takes ~30–60 s — several hot projection ops are
wrapped in `@torch.compile()` (`zeropower_via_polarexpress`,
`process_activation_target`, `matmul_proj_linf`, `matmul_proj`). After tracing
they're fast for the rest of the session.

If you're iterating locally and don't need full throughput, comment out the
`@torch.compile(...)` decorators on those functions in `src/ptorch/core/ops.py`
and `src/ptorch/optim_static.py`. Everything still runs in eager mode.

## Quick start (MNIST-shaped batch)

```python
import torch
import torch.nn.functional as F
import ptorch                          # applies projection overrides to torch
import ptorch.nn.modules as pnn
import ptorch.optim_static as poptim

x = torch.randn(8, 784)
y = torch.randint(0, 10, (8,))
y_onehot = F.one_hot(y, num_classes=10).float()

model = pnn.Linear(784, 10)
criterion = pnn.CrossEntropy()
optimizer = poptim.ProjectionSGD(model.parameters(), lr=1.0)

logits = model(x)
criterion(logits, y_onehot).sum().backward()   # propagates projection targets
optimizer.step()                                # p ← p − lr·(p − p_proj)
optimizer.zero_grad()
```

`loss.backward()` walks the graph and replaces each node's backward with its
projection (see [[Reference-Ops]]). The optimizer then converts the projection
target `p_proj` (held in `p.grad`) into the pseudo-gradient `p − p_proj`
before delegating to its inner `torch.optim` step (see
[[Reference-Optimizers]]).

## Smoke tests

Every config-driven benchmark accepts `--max-steps N --num-runs M`:

```bash
python -m experiments.mlp.bench_ptorch         --max-steps 10 --num-runs 1
python -m experiments.cnn_benchmarks.bench_ptorch --max-steps 10 --num-runs 1
python -m experiments.attention.bench_ptorch_vit  --max-steps  5 --num-runs 1
```

See [[Experiments-MLP]], [[Experiments-CNN]], [[Experiments-Attention]] for
the full per-benchmark walkthroughs.

## Where to go next

- New to the framework → [[Concepts-Feasibility-Framing]] +
  `mnist_from_scratch.ipynb` for a from-scratch walk-through.
- Building a model → [[Reference-NN-Modules]].
- Debugging a stalled run → [[Practical-Architecture-Init]] and
  [[Concepts-Vanishing-Targets]].
