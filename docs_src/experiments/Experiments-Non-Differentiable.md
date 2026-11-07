# Experiments: non-differentiable activations

**Thesis:** App. A.3 — *non-differentiable activation functions*.
**Directory:** `experiments/non_differentiable/`.

Trains MLPs whose activations have **no usable gradient anywhere** —
networks autograd literally cannot train end-to-end. This is the most
striking PTorch demonstration: projection targets propagate through `sign`,
`quantized ReLU`, and `gapped step` activations without any
straight-through estimator or surrogate gradient.

| Script | Purpose |
|---|---|
| `quantized_relu.py`   | Bench driver — sweeps activations Step / GappedStep / QuantizedRelu against ReLU baseline on MNIST |
| `plot_results.py`     | Accuracy curves per activation |

Config: `experiments/non_differentiable/config.yaml`.

## Activations

| Module (`pnn.`) | Backward op | Forward |
|---|---|---|
| `Step` | `StepProjection` | `+1` if `x ≥ 0` else `-1` |
| `GappedStep(delta=2.0)` | `GappedStepProjection` | `+1`/`-1` with a forbidden zone `(-δ/2, δ/2)` |
| `QuantizedRelu(step=1.0)` | `QuantizeReLUProjection` | `step · round(max(0, x) / step)` |

The forward outputs are constant almost everywhere, so the gradient is the
zero distribution. The projection-target backward instead picks the
**closest feasible point** on the forward graph and propagates that — see
[[Reference-Ops]] for the per-op derivation.

## Smoke test

```bash
python -m experiments.non_differentiable.quantized_relu \
    --config experiments/non_differentiable/config.yaml \
    --max-steps 10 --num-runs 1
python -m experiments.non_differentiable.plot_results
```

## How it maps to the thesis

- App. A.3 figures ↔ `plot_results.py` output. The PTorch curves should
  separate from random-chance early; baseline ReLU + autograd is included
  for sanity; baseline ReLU + a straight-through estimator can be added by
  swapping the activation in the bench script.

## See also

- [[Reference-Ops]] — `StepProjection`, `GappedStepProjection`,
  `QuantizeReLUProjection`.
- [[Reference-NN-Modules]] — `Step`, `GappedStep`, `QuantizedRelu`.
- [[Concepts-Feasibility-Framing]] — why the projection approach handles
  these naturally (the constraint set doesn't need a derivative, only a
  closed-form nearest-point oracle).
