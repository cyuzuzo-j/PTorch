# Reference: `core/overrides.py`

**File:** `src/ptorch/core/overrides.py` (270 LOC)
**Thesis:** §3.3 (graph primitives), App. D (closed-form projections for
`sum`/`add`/`mul`/`square`).

Importing `ptorch` runs `apply_overrides()` (see
`src/ptorch/__init__.py:3-5`), which monkey-patches a handful of `torch`
functions so that *every* operation in your model graph has a
projection-aware backward, even if you never reach for a `pnn.` module
directly.

## What gets patched

| Original | Replacement | Backward (projection mode) |
|---|---|---|
| `torch.sum`, `Tensor.sum` | `projected_sum` | `ProjectedSum.backward` — adds the residual `(z − sum(x))/(N+1)` uniformly |
| `torch.add`, `Tensor.add`, `Tensor.__add__`, `Tensor.__radd__` | `projected_add` | `ProjectedAdd.backward` — Lagrangian projection onto the hyperplane `x + α y = z` |
| `torch.mul`, `Tensor.mul`, `Tensor.__mul__` | `projected_mul` | `ProjectedMul.backward` — 5 Newton steps on the Hadamard constraint `x⋆ ⊙ y⋆ = z` |
| `torch.square`, `Tensor.square` | `projected_square` | `ProjectedSquare.backward` — `sign(x) · √max(z, 0)` |

The dispatchers `projected_*` short-circuit to the original implementations
whenever none of the operands `requires_grad` — i.e. there is **no** backward
overhead when projections are off or when the op runs on detached tensors
(see `overrides.py:191-244`).

## `apply_overrides()` *(overrides.py:246-257)*
Installs the four patches. Called automatically on `import ptorch`.

## `remove_overrides()` *(overrides.py:259-270)*
Restores the originals. Useful in tests that need to compare against vanilla
PyTorch.

## Why monkey-patching?

The cyclic-projection scheme of §3.2 requires *every* node in the autograd
graph to emit a projection target — not just the parametric layers. A plain
`x = a + b` in user code would otherwise silently fall back to the gradient
backward and confuse downstream layers' targets with gradients (a footgun
called out in App. B.1; see [[Practical-Architecture-Init]]). Patching the
free functions and `Tensor` dunder ops is the simplest way to enforce that
without asking users to rewrite their models with `pnn.Add(...)`.

## How it maps to the thesis

- `ProjectedSum` ↔ App. D linear constraint projection.
- `ProjectedAdd` ↔ §3.3 plane projection with Lagrangian normal `(1, α)`.
- `ProjectedMul` ↔ App. D bilinear projection, evaluated by 5 Newton steps.
- `ProjectedSquare` ↔ closed form from App. D for `y = x²`.

## First-run latency

`projected_sum`/`add`/`mul`/`square` are pure Python wrappers and don't go
through `@torch.compile`, so they don't add to the first-import cost. The
compile budget is paid by the **content** of their `.backward` (which calls
the compiled `matmul_proj*` and `process_activation_target`).

## See also

- [[Reference-Ops]] — `process_activation_target` is what
  `ProjectedAdd.backward` post-processes through.
- [[Practical-Architecture-Init]] — the "no torch/ptorch mixing" rule.
