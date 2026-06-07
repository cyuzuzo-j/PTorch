# Useful projections (catalogue)

**Thesis:** App. D.

A compact list of the closed-form projections used by PTorch, each one
pointed at the code that implements it. Use this page when you need to add
a new layer type and want to crib the projection scheme.

## Linear layers

### Bilinear equality `A B = Z`

- L₂-fused (consensus + bilinear): `matmul_proj` in
  `src/ptorch/core/ops.py:253-343`. Newton on a 1-D root equation (Elser
  surrogate, see code comment block at `ops.py:284-296`).
- L∞-fused: `matmul_proj_linf` in `ops.py:89-196`. Newton on the diamond
  radius `eps`.

### Linear hyperplane `x + α y = z`

- `ProjectedAdd.backward` in `src/ptorch/core/overrides.py:74-109`. Closed
  form via the Lagrange multiplier `t = (z − x − αy) / (1 + α²)`.

### Summation `Σx = z`

- `ProjectedSum.backward` in `overrides.py:21-60`. Add `(z − Σx)/(N+1)`
  uniformly.

### Hadamard product `x ⊙ y = z`

- `ProjectedMul.backward` in `overrides.py:111-162`. 5 damped Newton steps
  on the bilinear residual.

### Square `y = x²`

- `ProjectedSquare.backward` in `overrides.py:164-177`. Closed form
  `sign(x) · √max(z, 0)`.

## Functional layers

### RMS normalisation `z = x / RMS(x)`

- `RMSNormProjection.backward` in `ops.py:1058-1093`. Two-step closed form:
  `z̄ = √n · z / ‖z‖`, then `x̄ = mean(x · z̄) · z̄`.

### Softmax

- `SoftmaxProjection.backward` in `ops.py:593-627`. Step 1: simplex
  projection (positivity + L1 normalisation); Step 2: shift `log z̄` along
  the softmax-equivalence ray to be closest to `x` in L₂.

### MaxPool2D / global max

- `max_proj_pt_batch` in `ops.py:817-867`. Vectorised projection onto the
  max-graph: sort, compute candidate `z_k`, pick the candidate minimising
  squared distance.
- `MaxPool2DProjection` (`ops.py:919-981`) and `SeqMaxPoolProjection`
  (`ops.py:870-895`) reuse it.

### Average pool over a sequence dimension

- `SeqAvgPoolProjection.backward` in `ops.py:898-916`. Closed form: add
  `(z_target − mean(x))` uniformly across the pooled dimension.

### Conv2d patch consensus

- `ConvPatchProjection.backward` in `ops.py:995-1055`. Unfold → fold with
  per-pixel overlap-count normalisation.

## Activation functions

### ReLU two-branch

- `ReLUProjection` (`ops.py:563-590`) — L₂ distance pick.
- `ReLULInfinityProjection` (`ops.py:529-561`) — L∞ distance pick.

### LeakyReLU two-branch

- `LeakyReLUProjection` (`ops.py:666-697`). Branch 1: `y = α x` with
  `x ≤ 0`; closed form `x₁ = (x + α z)/(1 + α²)`. Branch 2: `y = x` with
  `x ≥ 0`; closed form `x₂ = (x + z)/2`.

### Step / sign

- `StepProjection` (`ops.py:700-721`). Keeps `x` where
  `sign(x) · z_target ≥ 0`, else snaps to `0`.

### Gapped step (dead zone)

- `GappedStepProjection(delta)` (`ops.py:774-814`). Picks the closer of
  `x_pos = max(x, δ/2)` (target `+1`) and `x_neg = min(x, −δ/2)` (target
  `−1`).

### Quantised ReLU staircase

- `QuantizeReLUProjection(step)` (`ops.py:723-772`). Tries the three
  staircase rungs around `round(z/step)` and picks the squared-distance
  minimiser.

### Causal-mask add `x + mask`

- `MaskedAddProjection` (`ops.py:629-646`). At `mask == −∞` positions, the
  target collapses to `x` so the per-position pseudo-gradient is zero.

### Branch consensus (multi-consumer)

- `BranchProjection(n_branches)` (`ops.py:648-662`). Returns
  `(sum of targets) / n_branches`.

## Cross-entropy proximal

`CrossEntropyProjection` (`ops.py:429-461`) is not a closed form — it is
`num_steps` fixed-point iterations of
`x ← x + λ(y − softmax(x))`, an approximation to the proximal operator
`prox_{λ·CE(·, y)}`. See [[Concepts-Cyclic-Projections]] for the role of
this proximal node in the cycle.

## See also

- [[Reference-Ops]] — same content, organised by file rather than by
  constraint shape.
- [[Concepts-Linear-Layer-Projection]] — the two big bilinear projections
  in detail.
