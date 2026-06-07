# Practical: optimizer and loss

**Thesis:** App. B.2, B.4.

## Optimizer choice

Recommended ordering, from App. B.2:

1. **`ProjectionMuon`** (or `ProjectionMuonV2` for distributed work) —
   default in `experiments/mlp/config.yaml` and the CNN benchmark. The
   orthogonalisation step matches the geometry of projection
   pseudo-gradients well.
2. **`ProjectionSGD(lr=1.0)`** — the "honest" baseline; recovers exact
   alternating projections, useful for parity tests.
3. **`ProjectionAdam(lr≈1.0)`** — works, but you **must** raise `lr` away
   from Adam's default `1e-3` (the docstring at `src/ptorch/optim_static.py:38`
   spells this out: pseudo-gradients are ~ proportional to parameter
   values, so an `lr` calibrated for gradient magnitudes is far too small).

`ProjectionAdagrad`/`ProjectionAdadelta` are included for completeness but
are not the headline recommendations.

The `optimizer_comparison` sweep in `experiments/mlp/` produces the
ablation figure behind these recommendations — see [[Experiments-MLP]].

## Projection mechanics knobs

`pnn.Linear` (and `pnn.Conv2D` through its inner Linear) expose:

- `alpha` — weight-vs-activation balance in the L₂ projection. Higher
  → moves weights more, activations less.
- `g` — output-target weight in the projection objective. Higher
  → projection respects the upstream target more strongly.
- `omega` — output scale; the forward returns `(A @ B) / omega`, the
  projection is computed against `Z · omega`. Used as a stability knob; rarely
  tuned manually.
- `num_iters` — Newton steps inside `matmul_proj` (L₂) or `matmul_proj_linf`
  (L∞). `1` is the production default for L₂; the L∞ branch usually wants
  `5`.

Global multipliers `config.projection_alpha` and `config.projection_g`
scale every layer's `alpha`/`g` at construction time — handy for global
sweeps without rewriting the model definition.

## Loss formulation

Three options ship with PTorch:

| Loss | Backward op | When to pick |
|---|---|---|
| `pnn.CrossEntropy` | `CrossEntropyProjection` (proximal, §3.2.2) | Default for classification |
| `pnn.HardMarginLoss(delta=1.0)` | `HardMarginProjection` | Margin classifier when you want exact boundary teleporting |
| `pnn.ProximalHingeMarginLoss(lambda_val=1.0)` | `ProximalHingeMargin` | Soft variant — bounded step toward the boundary |

### Tuning the CE proximal operator

`CrossEntropyProjection` exposes two knobs via the config:

- `config.cross_entropy_num_steps` (default `5`) — inner fixed-point
  iterations of `x ← x + λ(y − softmax(x))`. More iterations = closer to
  the true proximal point, at extra cost.
- `config.cross_entropy_lambda` (default `5.0`) — proximal strength. Larger
  λ = the loss "pulls harder" on the output activations; very large λ
  reduces to the hard label-equality constraint of §2.

The `effect_of_loss_config.yaml` benchmark in `experiments/mlp/` runs the
ablation that motivated these defaults.

## See also

- [[Reference-Optimizers]] — exact `step()` substitution.
- [[Reference-Ops]] — the `*Projection` ops behind each loss.
- [[Concepts-Cyclic-Projections]] — why the optimizer's `lr` is the
  *relaxation parameter* of §3.2.1.
