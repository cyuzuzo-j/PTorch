# Experiments: deep-network diagnostics

**Thesis:** Ch. 4 (vanishing-target theorem), §5.2 (depth sweep).
**Directory:** `experiments/deep/`.

Two scripts that produce the diagnostic figures behind the vanishing-target
analysis. Both are **single-file scripts with constants at the top** — no
YAML, no CLI flags — and you edit the constants to scale runs.

## `vanishing_target.py`

Sweeps network depth (typically 2 → 16) and logs, at every step, the L₂
norm of the projection target arriving at every layer. The thesis Fig.
in §5.2 plots `‖target_ℓ‖` vs ℓ for a handful of training steps; the curve
flattens to ~0 in the early layers as depth grows, which is the
projection-based analogue of vanishing gradients.

The script saves figures under `experiments/deep/images/`.

## `local_nonexpansiveness_deep.py`

Verifies the central lemma of Ch. 4 empirically. For each layer ℓ:

1. Take a real training activation distribution.
2. Compute the projection target `T(x)`.
3. Perturb the upstream target by random `δ` of controlled magnitude.
4. Measure `‖T(x + δ) − T(x)‖ / ‖δ‖` over many `δ`.

The empirical contraction ratio should sit in `[0, 1]` everywhere — which is
what the thesis proves theoretically.

Output: a contraction-ratio histogram per layer, again saved to
`experiments/deep/images/`.

## Running

```bash
python -m experiments.deep.vanishing_target
python -m experiments.deep.local_nonexpansiveness_deep
```

These can be slow at full sweep settings; halve the depth list at the top
of the file for a 1-minute smoke run.

## How it maps to the thesis

- Ch. 4 proof of vanishing targets ↔ `vanishing_target.py` depth curve.
- Ch. 4 non-expansiveness lemma ↔ `local_nonexpansiveness_deep.py` ratio
  histogram.
- §5.2 panel ↔ both figures combined (the thesis composes them into a single
  multi-panel figure — see `experiments/deep/images/` after running).

The practical consequence — *prefer shallow networks* — is restated in
[[Practical-Architecture-Init]].

## See also

- [[Concepts-Vanishing-Targets]] — theory.
- [[Practical-Architecture-Init]] — depth caps + initialisation heuristics.
- [[Experiments-MLP]] — `deep_mlp_config.yaml` is the training-side
  counterpart that shows accuracy collapsing with depth.
