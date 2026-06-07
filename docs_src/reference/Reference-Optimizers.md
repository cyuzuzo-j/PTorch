# Reference: `optim_static.py`

**File:** `src/ptorch/optim_static.py` (263 LOC)
**Thesis:** §3.2.1 (cyclic projections ↔ incremental GD), §3.2.2 (proximal
loss), App. B.2 (optimizer choice).

Every optimizer in this module wraps a standard `torch.optim` class and
performs **exactly one extra step before the parent's `.step()`**:

```python
# inside the optimizer's @torch.no_grad() step
for p in group['params']:
    if p.grad is not None:
        # p.grad currently holds the projection target p_proj
        # rewrite it as the pseudo-gradient (p − p_proj)
        p.grad.copy_(p.data - p.grad)
return super().step(closure)
```

That single substitution is what turns a projection target into a
gradient-style update. The parent optimizer's learning rate plays the role
of the **nonlinear relaxation parameter `α`** from §3.2.1 — `lr=1.0` recovers
exact alternating projections; `lr<1` produces a damped step.

## `class ProjectionSGD(params, lr=1.0, **kwargs)` *(optim_static.py:5-29)*
Wraps `torch.optim.SGD`. Default `lr=1.0`. Supports momentum + weight decay
through the kwargs.

## `class ProjectionAdam(params, **kwargs)` *(optim_static.py:31-53)*
Wraps `torch.optim.Adam`. Note from the docstring: PyTorch's `Adam` default
`lr=1e-3` is **too small** for projection pseudo-gradients (which are
roughly proportional to parameter values); raise it close to `1.0`.

## `class ProjectionAdagrad(params, **kwargs)` *(optim_static.py:55-72)*
## `class ProjectionAdadelta(params, **kwargs)` *(optim_static.py:74-91)*
## `class ProjectionMuon(params, **kwargs)` *(optim_static.py:94-111)*

All structurally identical to `ProjectionAdam`. `ProjectionMuon` is the
recommended baseline for the MLP/CNN benchmarks (the default in
`experiments/mlp/config.yaml`).

## `class ProjectionMuonV2(params, lr=1e-3, momentum=0.95, backend_steps=5, nesterov=True)` *(optim_static.py:186-263)*

An advanced variant that adds:

- **Distributed sharding** of the orthogonalisation work across ranks via
  `torch.distributed.all_reduce`.
- **Polar-Express orthogonalisation** of the pseudo-gradient (instead of
  Newton-Schulz5) via the in-file `zeropower_via_polarexpress`.
- An `_is_target` flag on parameters (when `config.use_hybrid=True`) that
  lets you mix projection-update and gradient-update parameters in the same
  optimizer — used by the hybrid optimization experiment (see
  [[Experiments-CNN]]).

## How it maps to the thesis

| Thesis claim | Code |
|---|---|
| §3.2.1: cyclic projection step ≡ incremental GD step | `p.grad.copy_(p.data - p.grad)` then parent `.step()` |
| §3.2.1: nonlinear relaxation by `α` | `lr` of the parent optimizer |
| App. B.2: Muon/Polar-Express for tall+thin or low-rank targets | `ProjectionMuonV2.zeropower_via_polarexpress` |

## See also

- [[Concepts-Cyclic-Projections]] — derivation of the GD ↔ projection
  identity.
- [[Practical-Optimizer-Loss]] — which optimizer to pick for which
  architecture.
- [[Reference-Ops]] — what populated `p.grad` in the first place.
