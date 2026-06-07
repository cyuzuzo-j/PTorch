# Experiments: ViT (attention)

**Thesis:** §5.4 — *Training a ViT*.
**Directory:** `experiments/attention/`.

| Script | Purpose |
|---|---|
| `bench_ptorch_vit.py` | PTorch ViT (uses `SoftmaxProjection`, `RMSNormProjection`, `MaskedAddProjection`, `BranchProjection`) |
| `bench_torch_vit.py`  | Vanilla PyTorch ViT baseline |
| `plot_results.py`     | Accuracy + loss curves |

All take `--config experiments/attention/config.yaml`; `--max-steps`,
`--num-runs` overrides supported.

## Why this experiment matters

A ViT exercises three projection ops that the MLP/CNN experiments don't:

- **`SoftmaxProjection`** (`src/ptorch/core/ops.py:593-627`) — mixed L₂/KL
  geometry; handles `-inf` from the causal mask correctly.
- **`MaskedAddProjection`** (`ops.py:629-646`) — replaces the target for
  masked positions with the original `x`, zeroing their pseudo-gradient.
- **`BranchProjection`** (`ops.py:648-662`) — averages targets from
  multiple consumers (residual connections, multi-head split) into a
  consensus target.
- **`RMSNormProjection`** (`ops.py:1058-1093`) — closed-form
  `x̄ = σ̄ · z̄` projection.

Together these let the projection cycle traverse a transformer block
unchanged.

## Smoke test

```bash
python -m experiments.attention.bench_ptorch_vit --max-steps 5 --num-runs 1
python -m experiments.attention.bench_torch_vit  --max-steps 5 --num-runs 1
python -m experiments.attention.plot_results
```

## How it maps to the thesis

- §5.4 accuracy curves ↔ `plot_results.py` output.
- The "constraint set per graph node" framing of §3.3 ↔ the
  `Softmax`/`MaskedAdd`/`Branch`/`RMSNorm` projection ops invoked inside the
  attention block.

## See also

- [[Reference-Ops]] — softmax, masked-add, branch, RMSNorm entries.
- [[Useful-Projections]] — App. D derivations.
