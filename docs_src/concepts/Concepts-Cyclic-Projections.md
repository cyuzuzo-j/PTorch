# Cyclic projections

**Thesis:** §3.2 (cyclic projections ↔ incremental GD, proximal loss node,
mini-batch extension).
**Code:** `src/ptorch/__init__.py` (auto-applies overrides),
`src/ptorch/core/overrides.py` (graph-walk surrogates),
`src/ptorch/optim_static.py` (relaxation step).

## What it does

Where Douglas–Rachford (PJAX) partitions the constraints in (P) into two
disjoint sets and bounces between them, PTorch instead **cycles through them
one at a time**, in an order that *coincides with the PyTorch autograd graph
walk*. The forward pass is unchanged; each node's backward emits a
projection target for its inputs; the optimizer's step is the *relaxation*
that converts those targets into a parameter update.

That choice has three consequences:

1. The partition is implicit — no 2-colouring step, no identity-node
   insertion, no per-batch graph surgery (compare PJAX §3 of the thesis).
2. Parameters are not duplicated per datapoint; the **fused** linear-layer
   projection of §3.1.1 handles consensus and bilinearity in one step (see
   [[Concepts-Linear-Layer-Projection]]).
3. Standard first-order optimizers double as *nonlinear relaxation* schemes
   (§3.2.1): `ProjectionSGD(lr=1.0)` recovers exact alternating projections;
   `ProjectionAdam` / `ProjectionMuon` add momentum and orthogonalisation on
   top of the projection target.

## Incremental gradient descent connection (§3.2.1)

The thesis proves that one cyclic projection over a single sample is
equivalent to one step of incremental gradient descent against a particular
quadratic surrogate. That equivalence is why the optimizer interface looks
exactly like the gradient-based one: `loss.backward()` populates `p.grad`
with `p_proj`, and `ProjectionSGD.step()` substitutes
`p.grad ← p.data − p.grad` before delegating to `torch.optim.SGD.step`. See
`src/ptorch/optim_static.py` lines 17–29 for the substitution.

## Proximal projections at the loss node (§3.2.2)

A "hard" output constraint (logits must equal one-hot labels) is too brittle
in practice. PTorch wraps the loss in a **proximal operator**:

`prox_{λL}(x₀) = argmin_x  λ · L(x, y) + ½‖x − x₀‖²`

For cross-entropy this is approximated by `num_steps` fixed-point iterations
of `x ← x + λ (y − softmax(x))` — see `CrossEntropyProjection.backward` in
`src/ptorch/core/ops.py`. `config.cross_entropy_num_steps` and
`config.cross_entropy_lambda` set the two knobs.

## Mini-batch extension (§3.2.3)

The cyclic order is extended over a mini-batch by averaging the per-sample
projection proposals before the relaxation step. This is what
`MatMulProjection.backward` does when it reshapes `(..., M, K) → (-1, K)` and
calls the fused projection on the batched flat tensor (see
`src/ptorch/core/ops.py` lines 364–410).

## How it maps to the thesis

| Thesis claim | Code |
|---|---|
| Backward = cyclic projection order | torch autograd walks the graph; each `autograd.Function.backward` returns a target |
| Optimizer = nonlinear relaxation | `ProjectionSGD.step` substitutes target → `(p − p_proj)`, then SGD |
| Proximal loss node | `CrossEntropyProjection` (5 inner steps by default) |
| Mini-batch averaging | per-sample reshape + average in `MatMulProjection.backward` |

## See also

- [[Concepts-Linear-Layer-Projection]] — what each linear projection inside
  the cycle is computing.
- [[Reference-Optimizers]] — the relaxation step in detail.
- [[Reference-Overrides]] — why `torch.sum/add/mul/square` need their own
  projection-aware backward.
