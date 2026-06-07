# Glossary

Terms used throughout the wiki, with the thesis section they are introduced
in.

**Feasibility problem** *(thesis §2.2, problem (P))*
Find a point in the intersection of multiple constraint sets, instead of
minimising a global loss. Training reformulated as: find network parameters
`θ` and intermediate activations `{hℓ}` lying jointly on every per-layer
constraint set.

**Constraint set** *(§2.2)*
The set of `(input, weights, output)` triples that satisfy one architectural,
data, or consensus equation. A linear layer's set is
`A = {(h, θ, h') | h' = θ h}`. See [[Concepts-Feasibility-Framing]].

**Projection operator** *(§2.2, §3.1)*
For a constraint set `C`, `Π_C(x)` returns the closest point in `C` to `x`.
The geometry is set by the norm (L₂ → `matmul_proj`, L∞ →
`matmul_proj_linf` in `src/ptorch/core/ops.py`).

**Target** *(§3.3)*
The projection result that flows backward in place of a gradient. Layer ℓ
receives `h⁽ℓ⁾` as a target, projects onto its constraint set, and emits a
target for layer ℓ−1.

**Consensus** *(§3.1)*
When multiple datapoints share parameters, their per-datapoint projections
must agree on a single `θ`. PJAX duplicates `θ` across the batch; PTorch
**fuses** consensus into the same step as the bilinear constraint (the
"fused" projection of §3.1.1).

**Cyclic projections** *(§3.2)*
Iteratively project onto each constraint set in a fixed order. The order
PTorch uses is *the order PyTorch already walks the autograd graph* — that is
why `loss.backward()` is enough to drive training.

**Non-expansive** *(§4)*
A map `T` is non-expansive if `‖T(x) − T(y)‖ ≤ ‖x − y‖`. PTorch's per-layer
target map is **locally** non-expansive, which is what makes targets shrink
as they propagate through depth — the *vanishing target* phenomenon (see
[[Concepts-Vanishing-Targets]]).

**Proximal operator** *(§3.2.2)*
`prox_{λL}(x₀) = argmin_x λ L(x, y) + ½‖x − x₀‖²`. PTorch uses a proximal
loss node so the loss couples *softly* to the output activations instead of
demanding exact match. Implemented for cross-entropy in
`CrossEntropyProjection` (`src/ptorch/core/ops.py`).

**Nonlinear relaxation** *(§3.2.1)*
Instead of jumping to the projected iterate, take a step *towards* it. With
relaxation parameter `α`, the update becomes `x ← (1−α)·x + α·Π_C(x)`. The
SGD-style optimizer's learning rate plays exactly this role (see
[[Reference-Optimizers]]).

**Incremental gradient descent connection** *(§3.2.1)*
Each cyclic projection step is shown to coincide with one step of incremental
gradient descent on a particular surrogate; that is why Adam/Muon-flavoured
relaxations work as drop-in optimizers over projection targets.

**Pseudo-gradient**
`p − p_proj`. What the optimizer consumes after `loss.backward()` populated
`p.grad` with a projection target. See `ProjectionSGD.step` in
`src/ptorch/optim_static.py`.
