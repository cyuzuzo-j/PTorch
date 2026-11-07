# Reference: `optim_static.py`

**File:** `src/ptorch/optim_static.py` (263 LOC)
**Thesis:** §3.2.1 (cyclic projections ↔ incremental GD), §3.2.2 (proximal
loss), App. B.2 (optimizer choice).

After `loss.backward()`, every parameter's `.grad` holds the pseudo-gradient
`p − p_proj` (see [[Reference-Ops]]; in gradient mode, the true gradient), so
every optimizer in this module is a standard `torch.optim` class with
projection-appropriate defaults, and there is nothing to convert. Linear, Conv2D,
Scale and Embedding form the residual in the solve: first-layer residuals sit
near the fp32 resolution of the weights, so `p − p_proj` computed from a stored
target would lose much of them. A Parameter used directly in a tensor op (`x * p`,
`h += p`) or as a module input gets it by that subtraction, as those ops return
targets. They only optimize `nn.Parameter`s: a plain leaf tensor is an activation
and its `.grad` holds a target.

A Parameter can also go through an op that only copies, selects or rearranges it,
with nothing to call by hand: `torch.cat` and `torch.stack` (and `hstack`, `vstack`,
`dstack`, `column_stack`), indexing, `clone`, casts and device moves (`.to`, `.double`,
`.half`, `.cuda`, ...), `reshape`, `flatten`, `contiguous`, `repeat`, `tile`,
`repeat_interleave`, `flip`, `roll`, `rot90`, `index_select`, `gather`, `take`,
`take_along_dim`, `masked_select`, `masked_fill`, `torch.where`, `tril`, `triu` and
`F.pad`. The op stays native and its output is converted, so `.grad` gets
`Jᵀ(y − z)` for a target `z` on the output `y`: repeated coordinates add up and
unselected ones stay at 0. Views (`view`, `narrow`, `split`, `expand`, ...) need
nothing either, as long as they then enter one of these ops, a tensor op or a module.
The list is in `src/ptorch/core/overrides.py`. All of this acts only when a Parameter
is involved, in projection mode with grad enabled; activations keep the native ops.

Still not covered, and silent, so the raw target or something derived from it reaches
`.grad`:

- `F.embedding` and `nn.Embedding`, on purpose. `nn.Embedding` + `Conversion` passes a
  true gradient down, which a conversion would turn back into the target. Use
  `Embedding` from `modules_experimental` or `p[ids]`.
- Writes of a Parameter into another tensor: `x[i] = p`, `x.copy_(p)`, `index_add`,
  `index_put`, `scatter`, `masked_scatter`.
- `backward()` started directly on a view of a Parameter or of a copy of one.
- A partial selection of `torch.cat([p, activation])` and the like: an op that mixes a
  Parameter with an activation counts as an activation from then on.
- Ops that change values (`p.exp()`, `torch.tanh(p)`, `p @ x`, `p / 2`, `-p`, `p.mean()`,
  `F.linear(x, p)`). A target has no meaning through their chain rule. For a true
  gradient of `½|f(p) − z|²`, put `Conversion` on the output: `Conversion()(p.exp())`.

Below a `Conversion` bridge true gradients flow, so a covered op on a Parameter there
converts a gradient and gives a wrong `.grad`: `x + p` already did, and `p.clone()` or
`p[idx]` do too. The layers used below a bridge in this repository (`nn.Conv2d`,
`nn.Embedding`) apply none of these ops to their weights.

The parent optimizer's learning rate plays the role
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
`experiments/neurips/mlp/config.yaml`).

> **STATUS (2026-07-12): Muon confirmed load-bearing, now with theory.** Projection
> transport is correct only up to positive per-path scalars (scalar-freedom theorem);
> a normalizing optimizer absorbs them, which is why `MUON_SHAPE_SCALE=1` appears in
> every winning GPT stack. See `experiments/parameter-golf/transport_theory.md` §4 and
> `STATUS_LEDGER.md`.

## How it maps to the thesis

| Thesis claim | Code |
|---|---|
| §3.2.1: cyclic projection step ≡ incremental GD step | `.grad = p − p_proj`, parent `.step()` |
| §3.2.1: nonlinear relaxation by `α` | `lr` of the parent optimizer |

## See also

- [[Concepts-Cyclic-Projections]] — derivation of the GD ↔ projection
  identity.
- [[Practical-Optimizer-Loss]] — which optimizer to pick for which
  architecture.
- [[Reference-Ops]] — what populated `p.grad` in the first place.
