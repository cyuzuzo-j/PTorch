# Vanishing targets

> **STATUS (2026-07-12): theorem valid; the cure story changed.** The theorem is now
> understood as a *dissipation law* (which channels eat the signal:
> `experiments/parameter-golf/vanishing_targets_theory.md`), and the TD(λ) remedy
> described below is a positive-scalar amplitude compensator — exact only on isotropic
> serial chains (deep-MLP wins are real), provably unable to fix the directional/fan-in
> defect that dominates transformers (scalar-freedom theorem,
> `experiments/parameter-golf/transport_theory.md` §4). The transformer fix is merge-ρ
> fan-in equalization (`MERGE_RHO`, no-VJP 3.2465 vs grad 3.1028). Record stacks run
> `TD_LAMBDA=0`. Verdicts: `experiments/parameter-golf/STATUS_LEDGER.md`.

**Thesis:** Ch. 4 — *vanishing target* theorem and local non-expansiveness
diagnostic.
**Code:** `experiments/neurips/deep/vanishing_target.py`,
`experiments/deep/local_nonexpansiveness_deep.py`.

## What it does

This chapter is the projection-based analogue of vanishing gradients. The
thesis proves that the per-layer target map of the cyclic-projection scheme
is **locally non-expansive**: small target deviations at layer `ℓ` shrink as
they propagate backward to layer `ℓ−k`. Empirically the consequence is a
training signal that decays with depth and a plateau in accuracy past
~4 layers in the shallow-MLP regime.

Two scripts (`experiments/neurips/deep/vanishing_target.py`, `experiments/deep/local_nonexpansiveness_deep.py`) instrument this directly:

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
python -m experiments.neurips.deep.vanishing_target --config experiments/neurips/deep/vanishing_target_config.yaml
python -m experiments.deep.local_nonexpansiveness_deep
```

Figures save into `experiments/deep/images/`. The
[[Experiments-Deep-Diagnostics]] page lists the exact filenames.

## Current understanding (2026-07-12)

The amplitude theorem above **stands** — it is a real dissipation law, numerically
confirmed. But the GPT parity campaign showed the *practical* disease is
**directional/metric, not amplitude**: displacement transport responds to an op's
singular value σ as `1/σ` (achievability) where gradient transport responds as σ
(sensitivity) — a `σ²` per-direction mis-weighting per hop.

- On serial **chains** this is harmless: the error is a positive per-path scalar,
  absorbed by any normalizing optimizer (scalar-freedom theorem; Muon absorbs scalars).
- At residual **fan-ins** it is fatal: the skip and sublayer paths accumulate
  *different* scalars, so the merged direction is destroyed (measured 10³–10⁴
  relative path mis-weighting at GPT residual merges).
- The fix is **local fan-in equalization** (merge-ρ: weight each branch by an EMA
  estimate of its squared gain before combining). No-VJP record **3.2465** vs the
  gradient twin's **3.1028** — no covector chain needed.

This also explains why amplitude-only repairs (TD(λ), floors, caps, line search)
fixed deep MLPs but never GPTs: positive scalars provably cannot repair a relative
path mis-weighting. Full treatment:
`experiments/parameter-golf/transport_theory.md` and
`experiments/parameter-golf/vanishing_targets_theory.md`; scoreboard:
`experiments/parameter-golf/STATUS_LEDGER.md`.

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
