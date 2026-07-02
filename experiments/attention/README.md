# Attention Experiment

Mini Vision Transformer benchmarks comparing projection-based training
(`bench_ptorch_vit.py`) against a standard gradient baseline
(`bench_torch_vit.py`) on MNIST/CIFAR-10.

## Files
- `config.yaml`: shared configuration (dataset, model size, optimizers).
- `bench_ptorch_vit.py`: projection ViT with ablation flags (below).
- `bench_torch_vit.py`: standard PyTorch ViT + AdamW reference.
- `diag_targets.py`: target-health diagnostics — per-layer relative target
  delta `||t - a|| / ||a||` through depth, softmax clamp/displacement stats,
  matmul Lagrange-multiplier stats, RMSNorm sigma flips. Accepts the same
  flags as the bench so each fix can be re-measured:
  `python -m experiments.attention.diag_targets --steps 300 [flags]`
- `plot_results.py`: joins the per-run CSVs in `results/`.
- `probe_head.py`: linear-probe diagnostic — freezes a projection-trained body
  (saved with `--save-best`), trains a plain AdamW linear readout on its
  features. Separates "body features are weak" from "the head solve is the
  ceiling": `python -m experiments.attention.probe_head --checkpoint results/ckpt_<tag>_run1.pt`
  (control: `--random-body`).

## Ablation flags (bench_ptorch_vit.py and diag_targets.py)

All defaults reproduce the legacy behavior; the CSV/file tag encodes
non-default flags.

| flag | values | what it does |
|---|---|---|
| `--prenorm` | `none`/`rms`/`dyt` | pre-norm wiring of each block (`none` = historical behavior; earlier revisions computed the norm but never consumed it) |
| `--branch-mode` | `mean`/`delta_sum` | fan-out target combination; `delta_sum` = backprop analog, keeps gain-1 skip paths |
| `--softmax-mode` | `legacy`/`rel_floor`/`simplex_l2` | attention-weight target handling (rel_floor bounds log-target displacement) |
| `--qk-match-g` | flag | sets g=omega in the QK^T solve so the 1/sqrt(d) scale stops weakening q/k targets |
| `--rmsnorm-mode` | `legacy`/`exact` | exact joint projection onto the RMSNorm graph manifold |
| `--residual-mode` | `full`/`half` | delta routing in ResidualAdd (half = strict min-norm split) |
| `--muon-acts` (+`--muon-acts-lr/-eps/-mode/-max-dim`) | flag | Activation residual rescaling. `--muon-acts-mode` = `fixed` (legacy: polar × constant lr — fails everywhere), `rel_row` (polar × lr·\|\|A row\|\|, body-layer fix), `rel_frob` (whole-tensor variant), `raw_rel_row` (skip polar, only rescale magnitude). `--muon-acts-max-dim` (default 4096) skips the rescale for the flatten head whose 8192-dim rows would yield O(1) head-target overshoot. |
| `--muon-acts-decay` (+`--muon-acts-decay-steps`) | `none`/`cosine`/`linear` | anneal `muon_acts_lr` to 0 over the decay horizon (0 = max_steps). The rescale is an early-training bootstrap; left constant it compounds noise past the ~9k peak (train-acc collapse in the 30k rel_row run) |
| `--momentum` | float (0.95) | ProjectionMuonV2 momentum (lower = less integration of polar direction-noise) |
| `--weight-decay` | float (0.0) | ProjectionMuonV2 decoupled weight decay; bounds the \|\|row(A)\|\|-growth feedback that rel_row amplifies (AdamW reference uses 0.01) |
| `--ce-lambda` / `--ce-num-steps` | float (1.0) / int (5) | CrossEntropyProjection proximal step size / inner iterations |
| `--lr-schedule` | `constant`/`cosine` | optimizer weight-lr schedule |
| `--save-best` | flag | save the best-val state to `results/ckpt_<tag>_run<n>.pt` for `probe_head.py` (not part of the flag tag) |
| `--proj-alpha` | float or `auto` | weight-vs-activation balance of bilinear solves; `auto` = qa/qb per layer |
| `--proj-g` | float | target-fidelity weight of bilinear solves |
| `--head-pool` | `flatten`/`avg`/`max` | classifier readout (flatten = 8192-dim head rows) |

## Key diagnostic finding (2026-06)

With all-default flags the transformer body receives **no** teaching signal:
the relative target delta is ~1e-8 (float32 noise) at every block while the
head sees ~3e-2. The exact bilinear projection routes essentially the whole
constraint displacement into the *weights* (qa = ||activation row||^2 ≈ 3e4
vs qb = ||weight col||^2 ≈ 2 at the flatten head), so the backward activation
target dies at the first solve — a per-layer form of the vanishing-target
theorem. `--proj-alpha auto` rebalances the solve (t becomes healthy) but the
min-norm displacement is still |B|-scaled; `--muon-acts` restores per-layer
magnitude (~1e-4..1e-3 body deltas) at the cost of replacing real targets
with fixed-size steps.
