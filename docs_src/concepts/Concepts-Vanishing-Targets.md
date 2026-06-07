# Vanishing targets

**Thesis:** Ch. 4 — *vanishing target* theorem and local non-expansiveness
diagnostic.
**Code:** `experiments/deep/vanishing_target.py`,
`experiments/deep/local_nonexpansiveness_deep.py`.

## What it does

This chapter is the projection-based analogue of vanishing gradients. The
thesis proves that the per-layer target map of the cyclic-projection scheme
is **locally non-expansive**: small target deviations at layer `ℓ` shrink as
they propagate backward to layer `ℓ−k`. Empirically the consequence is a
training signal that decays with depth and a plateau in accuracy past
~4 layers in the shallow-MLP regime.

The two scripts in `experiments/deep/` instrument this directly:

- **`vanishing_target.py`** — runs a depth sweep and logs the norm of the
  target vector at every layer, every step. Produces the depth-vs-signal
  curve from §4 / §5.2 of the thesis.
- **`local_nonexpansiveness_deep.py`** — perturbs the input target by `δ`,
  measures the perturbation magnitude after one layer's backward, and reports
  the empirical Lipschitz constant. Confirms `‖T(x+δ) − T(x)‖ ≤ ‖δ‖` per
  layer on real activation distributions.

Both scripts use **hardcoded constants at the top of the file**; edit them
to scale runs up/down (called out in `README.md`).

## How to use

```bash
python -m experiments.deep.vanishing_target
python -m experiments.deep.local_nonexpansiveness_deep
```

Figures save into `experiments/deep/images/`. The
[[Experiments-Deep-Diagnostics]] page lists the exact filenames.

## How it maps to the thesis

- §4 proof — the contraction factor `q < 1` of the target map.
- §5.2 figure — depth × step × target-norm grid produced by
  `vanishing_target.py`.
- App. B.1 (shallow networks preferred) is the practical takeaway —
  see [[Practical-Architecture-Init]].

## See also

- [[Experiments-Deep-Diagnostics]] — what to look at, what to expect.
- [[Practical-Architecture-Init]] — depth caps and init heuristics that
  follow from this analysis.
