# Reference: `core/ops.py`

> **STATUS (2026-07-12): structurally correct, a campaign behind.** `ops.py` is now
> ~2379 LOC (was 1109 when this page was written) — line references are stale, search by
> class name. Ops gained multiple backward *modes* during the 2026-07 GPT campaign
> (e.g. `SOFTMAX_MODE=jacobian` and `ATTN_VALUE_ADJOINT=1` are the working defaults;
> legacy softmax log modes, `RMSNORM_MODE` legacy, softcap-atanh, and `SCALE_ADJOINT`
> are falsified). Per-mode verdicts: docstrings in `ops.py` and
> `experiments/parameter-golf/STATUS_LEDGER.md`.

**File:** `src/ptorch/core/ops.py` (1109 LOC)
**Thesis:** §3.1 (linear/bilinear), App. D (closed-form projections).

Every entry below is a `torch.autograd.Function`. Its `.forward` runs the
ordinary op so the forward pass is identical to standard PyTorch; its
`.backward` returns a **projection target** instead of a gradient.

## Helpers

### `zeropower_via_polarexpress(G, steps=5, eps=1e-2)` *(ops.py:24-62)*
Computes the polar factor of `G` via the Polar-Express polynomial iteration —
drop-in replacement for Newton-Schulz5. Used both as a relaxation
post-processor (`process_activation_target`).
Wrapped in `@torch.compile()`.

### `process_activation_target(A_det, A_proj)` *(ops.py:64-83)*
Post-processes an activation projection by orthogonalising the
"pseudo-gradient" `A − A_proj` with `zeropower_via_polarexpress` when
`config.use_muon_activations` is on. Otherwise returns `A_proj` unchanged.
Controlled by `config.muon_activations_lr`.

## Linear / bilinear projections

### `matmul_proj(A, B, Z, *, alpha, g, omega, num_steps, residual=False)` *(ops.py:253-343)*
Fused L₂ projection onto `A B = Z` (or `A − AB = Z` if `residual`).
Implements the bracketed-Newton root-finder of §3.1.1 (Elser-style surrogate
equation). Returns `(A_proj, B_proj, Z_proj, t)` where `t` is the warm-start
Lagrange scalar to cache.

> **STATUS (2026-07-12): confirmed keeper.** The weight update is provably a
> diagonally-GN-preconditioned gradient step on the local misfit (cos 1.000000 to
> closed form) — transport-agnostic; it adds curvature information backprop lacks.
> A converged-entry teleport bug in the root-finder was fixed 2026-07-07. See
> `experiments/parameter-golf/transport_theory.md` §2 and `STATUS_LEDGER.md`.

### `class MatMulProjection(autograd.Function)` *(ops.py:347-410)*
Wraps `matmul_proj` for autograd. Forward is `(A @ B) / omega`. Backward
receives the upstream target, reshapes `(N..., M, K) → (-1, K)` for
mini-batch averaging, calls `matmul_proj`, updates the cache, and returns
the projection targets for `A` and `B`. Final post-processing through
`process_activation_target`.

### `matmul_proj_linf(A, B, Z, *, eps_init, g, omega, num_steps)` *(ops.py:89-196)*
Exact L∞ (Chebyshev) version. Newton on the diamond-radius `eps` of §3.1.2;
analytic consensus reconstruction. Returns `(A_proj, B_proj, Z_proj, eps)`.

### `class MatMulProjectionLinf(autograd.Function)` *(ops.py:199-251)*
Autograd wrapper analogous to `MatMulProjection` but for the L∞ branch.

## Activation projections

### `class ReLUProjection` (L₂) *(ops.py:563-590)*
Two-branch projection: inactive (`x ≤ 0, y = 0`) vs active (`x ≥ 0, y = x`).
Picks the branch with the smaller squared distance.

### `class ReLULInfinityProjection` *(ops.py:529-561)*
Same two-branch decomposition under L∞ distance.

### `class LeakyReLUProjection(negative_slope=0.01)` *(ops.py:666-697)*
Two-branch projection accounting for the `slope · x` inactive branch.

### `class StepProjection` *(ops.py:700-721)*
Backward for the hard step `sign(x)`. Keeps `x` where `sign(x) · z_target ≥ 0`;
otherwise pushes `x` to `0`. The non-differentiable activation showcase.

### `class GappedStepProjection(delta=2.0)` *(ops.py:774-814)*
Step activation with a dead zone `(-δ/2, δ/2)`. Picks the closer of the two
feasible branches.

### `class QuantizeReLUProjection(step=1.0)` *(ops.py:723-772)*
Quantized ReLU. Projects onto the nearest staircase rung; tries `k₀ ∈
{−1, 0, +1}` candidates around the nearest target rung.

### `class SoftmaxProjection` *(ops.py:593-627)*
Mixed L₂/KL geometry — projects `z` onto the simplex (positivity + L1
normalisation), then shifts `log z̄` to the closest point on the
softmax-equivalence ray of `x`. Handles `-inf` entries from causal masks.

> **STATUS (2026-07-12): superseded for transformer transport by
> `SOFTMAX_MODE=jacobian`** (exact own-op Jᵀ). The log-space simplex retarget is a
> `1/σ` preimage rule — its logit target measures cos ~0.08 to the true adjoint and
> kills q/k learning. Falsified for the GPT stack; see
> `experiments/parameter-golf/STATUS_LEDGER.md`.

## Pooling / consensus

### `max_proj_pt_batch(a, z)` *(ops.py:817-867)*
Vectorised projection onto the maximum function graph. Used by
`MaxPool2DProjection` and `SeqMaxPoolProjection`.

### `class MaxPool2DProjection` *(ops.py:919-981)*
Backward for `max_pool2d`. Unfolds the input into patches, runs
`max_proj_pt_batch` per patch, folds the result back.

### `class SeqMaxPoolProjection` *(ops.py:870-895)*
GNN-style global max-pool over the token dimension `(B, T, D) → (B, D)`.

### `class SeqAvgPoolProjection` *(ops.py:898-916)*
Closed-form linear-constraint correction: add `(z_target − mean(x))`
uniformly across `T`.

### `class ConvPatchProjection` *(ops.py:995-1055)*
Handles the spatial consensus inside `Conv2D`. Unfold → linear projection →
fold with overlap-count normalisation.

### `class RMSNormProjection` *(ops.py:1058-1093)*
Closed-form input projection from §App. D: `z̄ = √n · z / ‖z‖`,
`σ̄ = mean(x · z̄)`, `x̄ = σ̄ · z̄`.

> **STATUS (2026-07-12): legacy renormalize backward falsified for GPT transport** —
> it sign-flips and fully re-amplifies vanished targets (`rms²` mis-weighting per
> crossing). Superseded by `RMSNORM_MODE=tangent` (off-VJP stacks) / `adjoint` (with a
> VJP chain); `adjoint` standalone off-VJP is itself falsified (3.9768, scale drift).
> See `experiments/parameter-golf/STATUS_LEDGER.md`.

## Loss projections

### `class CrossEntropyProjection` *(ops.py:429-461)*
Proximal operator `prox_{λ·CE(·, y)}` evaluated by `num_steps` fixed-point
iterations of `x ← x + λ(y − softmax(x))`. Defaults from `config`:
`num_steps=5`, `λ=5.0`.

> **STATUS (2026-07-12):** `num_steps=5` confirmed keeper off-VJP; `num_steps=1`
> (exact ∇CE seed) helps only with a transitive-VJP chain (+0.25 falsified without).
> See [[Reference-Config]] and `experiments/parameter-golf/STATUS_LEDGER.md`.

### `class MSEProjection` *(ops.py:413-427)*
Closed form: average of predictions and targets.

### `class HardMarginProjection` *(ops.py:463-483)*
Snaps logits to the margin boundary in one step. The "strict teacher"
variant.

### `class ProximalHingeMargin` *(ops.py:486-513)*
Bounded-step variant: moves logits towards the boundary by at most
`λ_val`.

## Other graph primitives

### `class MaskedAddProjection` *(ops.py:629-646)*
Backward-aware causal mask add: replaces the target for `mask == −∞`
positions with the original `x` so masked positions get zero pseudo-gradient.

### `class BranchProjection` *(ops.py:648-662)*
Consensus across multiple consumers — averages targets from `N` branches
back into one.

> **STATUS (2026-07-12):** the plain mean is a **metric-blind merge** — fan-ins/outs
> are where projection transport fails (paths arrive with scalar weights off by
> 10³–10⁴; measured merged cos-to-gradient 0.04). `delta_sum` mode alone was a void
> (falsified 2026-07-10): no combine rule fixes inconsistent inputs. Fixed by
> **merge-ρ path-scalar equalization** (`MERGE_RHO`, no-VJP record 3.2465). See
> `experiments/parameter-golf/transport_theory.md` §4/T2 and `STATUS_LEDGER.md`.

### `class Conversion` *(ops.py:1097-1108)*
Adapter: converts a projection target back into a gradient
(`grad = input − z_target`) at the boundary with non-projection-aware code.

## See also

- [[Reference-NN-Modules]] — `nn.Module` wrappers that call these.
- [[Concepts-Linear-Layer-Projection]] — the theory behind `matmul_proj*`.
- [[Useful-Projections]] — catalogue of closed forms with thesis derivations.
