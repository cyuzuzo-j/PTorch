# Reference: `nn/modules.py`

**File:** `src/ptorch/nn/modules.py` (303 LOC)
**Thesis:** §2.2 (constraint sets), §3.3 (per-layer projections).

Importable as `import ptorch.nn.modules as pnn`. Every class is a
`nn.Module` subclass that wraps an `autograd.Function` from
[[Reference-Ops]]. If `config.use_projections` is False they fall back to
standard PyTorch ops, which is handy for parity tests.

## Base

### `class ProjectionModule(outputs=0)` *(modules.py:8-13)*
Tracks a `projection_forward_cache` list of length `outputs`, used by ops
that need to remember the projected forward output (e.g. `ReLUProjection`
stashes the chosen branch).

## Linear / convolution

### `class Linear(in_features, out_features, *, bias=True, alpha=1.0, g=1.0, omega=1.0, num_iters=1, norm='l2', use_cache=True)` *(modules.py:15-80)*
Dispatches to `MatMulProjection` (`norm='l2'`) or `MatMulProjectionLinf`
(`norm='linf'`). Bias is folded into the projection by appending a constant
`1` column to the input (lines 62-65). `proj_cache` warm-starts the Lagrange
scalar across iterations. `alpha`, `g` multiplied by `config.projection_alpha`
and `config.projection_g` at construction.

### `class Conv2D(in_channels, out_channels, kernel_size=3, stride=1, padding=0, *, bias=True, alpha=1.0, g=1.0, num_iters=5)` *(modules.py:219-290)*
Implements convolution as **unfold → fused linear projection → fold**, with
spatial consensus handled by `ConvPatchProjection` in the unfold step. The
inner `Linear(use_cache=False)` runs a flat global consensus across batch +
spatial dims, avoiding the `O(MNK)` blow-up of a naive per-patch projection
(see comment at `modules.py:248-250`). Supports `padding='same'`,
`padding='valid'`, or explicit tuples.

### `class MaxPool2d(kernel_size, stride=None, padding=0)` *(modules.py:293-303)*
Wraps `MaxPool2DProjection`.

## Activations

### `class ReLU(norm='l2')` *(modules.py:82-93)*
Dispatches to `ReLULInfinityProjection` or `ReLUProjection` depending on
`norm`.

### `class Softmax()` *(modules.py:95-107)*
Wraps `SoftmaxProjection`.

### `class LeakyReLU(negative_slope=0.01, inplace=False)` *(modules.py:110-117)*
Subclasses `nn.LeakyReLU`; routes backward through `LeakyReLUProjection`
when projections are on.

### `class Step()` *(modules.py:120-129)*
Non-differentiable `sign`-style activation. Forward returns ±1; backward
uses `StepProjection`.

### `class GappedStep(delta=2.0)` *(modules.py:147-175)*
Step with a forbidden zone `(-δ/2, δ/2)`. Forward returns ±1; backward via
`GappedStepProjection`.

### `class QuantizedRelu(step=1.0)` *(modules.py:131-145)*
Forward applies `step · round(max(0, x) / step)`; backward via
`QuantizeReLUProjection`.

## Losses

### `class CrossEntropy()` *(modules.py:178-185)*
Backward applies `CrossEntropyProjection` (proximal operator). When
projections are off, falls back to `F.cross_entropy`.

### `class HardMarginLoss(delta=1.0)` *(modules.py:187-197)*
Strict-margin classifier loss. Projection backward via `HardMarginProjection`;
gradient fallback returns the same hinge-style mean-squared error.

### `class ProximalHingeMarginLoss(lambda_val=1.0)` *(modules.py:200-216)*
Soft variant: takes a bounded step (size `λ`) toward the margin boundary
instead of teleporting.

## Constraint-set ↔ module map (§3.3 walk)

| Constraint set | Module | Backward op |
|---|---|---|
| `Aℓ_lin = {(h, θ, h') | h' = θ h}` | `Linear` | `MatMulProjection(Linf)` |
| `Aℓ_relu = {(x, y) | y = ReLU(x)}` | `ReLU` | `ReLU{,LInfinity}Projection` |
| `Aℓ_conv = {(x, θ, y) | y = conv(x, θ)}` | `Conv2D` | `ConvPatchProjection` + `Linear` |
| `Aℓ_maxpool` | `MaxPool2d` | `MaxPool2DProjection` |
| Loss `L_out = {h_L = y}` | `CrossEntropy` / `HardMarginLoss` | `CrossEntropyProjection` etc. |

## See also

- [[Reference-Ops]] — the underlying `autograd.Function`s.
- [[Reference-Config]] — runtime knobs (`use_projections`,
  `cross_entropy_*`, `projection_alpha`, …).
- [[Concepts-Feasibility-Framing]] — what these constraint sets are.
