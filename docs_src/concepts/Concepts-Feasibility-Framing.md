# Feasibility framing

**Thesis:** Ch. 2.
**Code:** `mnist_from_scratch.ipynb` (clean instantiation),
`src/ptorch/nn/modules.py` (per-layer constraint sets).

## What it does

Recasts neural-network training from "minimise a global loss" into "find a
point `(θ, {hℓ}, y)` that lies on every per-layer constraint set
simultaneously" — the *feasibility problem* (P) of §2.2. There is no global
gradient: each layer owns a constraint, each constraint owns a projection,
and the optimizer is just the rule for sweeping through them.

## The constraint sets

For an L-layer network, the thesis defines (§2.2):

- **Architectural** `Aℓ = {(h⁽ℓ⁻¹⁾, θ⁽ℓ⁾, h⁽ℓ⁾) | h⁽ℓ⁾ = fℓ(h⁽ℓ⁻¹⁾, θ⁽ℓ⁾)}`
  — every layer's forward equation. Implemented by the per-module backward
  in `src/ptorch/nn/modules.py` (Linear → `MatMulProjection.backward`,
  ReLU → `ReLUProjection.backward`, …).
- **Data** `D = {(h⁽⁰⁾, y) | (h⁽⁰⁾, y) is a training sample}`.
- **Loss/output** — equality between final activation and label, optionally
  *softened* by a proximal node (§3.2.2). See
  `CrossEntropyProjection.backward` in `src/ptorch/core/ops.py`.

PTorch's contribution is to make the *order* in which these are visited match
PyTorch's autograd walk, so the cyclic projection algorithm "is" the backward
pass — covered in [[Concepts-Cyclic-Projections]].

## XOR walkthrough (§2.3)

The thesis builds intuition with a 2-layer XOR MLP whose constraint sets are
small enough to visualise (Fig. 2.2 in the PDF). Each forward pass produces a
point; each backward pass projects it back onto the closest feasible point;
iterating converges to a solution. The same flow drives every later
experiment — only the constraint sets get richer.

For a fully worked, dependency-free reproduction read
`mnist_from_scratch.ipynb`. It implements the same projection rules manually
on top of NumPy/PyTorch tensors, so you can watch the targets propagate
without the framework layer.

## How it maps to the thesis

- §2.2 problem (P) ↔ the union of constraint sets above.
- §2.3 XOR figures ↔ the per-layer projections in
  `src/ptorch/core/ops.py` evaluated on a 2-D toy.
- The handoff to a *scalable* solver (consensus + bilinear fusion) is taken
  up in [[Concepts-Linear-Layer-Projection]] and the proof that cyclic
  projections drive training is in [[Concepts-Cyclic-Projections]].

## See also

- [[Concepts-Cyclic-Projections]]
- [[Concepts-Linear-Layer-Projection]]
- [[Reference-NN-Modules]]
