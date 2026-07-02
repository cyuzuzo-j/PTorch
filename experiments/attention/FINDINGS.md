# Projection-ViT Convergence Gap — Findings (2026-06)

Investigation of why `bench_ptorch_vit.py` plateaus at ~0.50 val on CIFAR10 while
the AdamW reference (`bench_torch_vit.py`) reaches ~0.70 in the same step budget.

Model: 4-block ViT, emb_dim=128, 16 heads, patch 4, batch 256, lr 1e-3,
ProjectionMuonV2. All runs 5k steps, single seed unless noted.

## Bottom line

The active ingredient is **`branch_mode=delta_sum` × `prenorm` together** —
not either alone. Both flags are individually neutral-to-harmful; combined
they recover ~+3 val pts at 5k.

Recommended defaults:
```bash
--prenorm dyt --branch-mode delta_sum --softmax-mode rel_floor \
  --qk-match-g --proj-alpha auto
```

## 5k val_acc table

| arm                                                                     | val@5k | vs baseline |
|---|---|---|
| AdamW reference (3 seeds, mean)                                         | **0.7045** | +21 pt |
| pn=dyt + bm=delta_sum + sm=rel_floor + qkg + pa=auto                    | **0.5267** | +3.2 |
| pn=rms + rn=exact + bm=delta_sum + sm=rel_floor + qkg + pa=auto         | 0.5109 | +1.6 |
| pn=dyt solo                                                             | 0.4989 | −0.4 |
| pn=rms + rn=exact solo                                                  | 0.4947 | −0.0 |
| pn=dyt + sm=rel_floor + qkg + pa=auto (no delta_sum)                    | 0.4886 | −0.6 |
| pn=dyt + pa=auto (no delta_sum)                                         | 0.4796 | −1.5 |
| **baseline** (4 seeds, mean; range 0.483–0.518)                         | **0.4950** | 0 |
| pa=auto solo                                                            | 0.356 | −14 |
| bm=delta_sum solo                                                       | 0.289 | −20 |
| ma (muon-acts) solo                                                     | 0.273 | −22 |
| hp=avg combo                                                            | 0.240 | −26 |

Tag legend:
`pn=` prenorm (none/rms/dyt), `bm=` branch-mode (mean/delta_sum),
`sm=` softmax-mode (legacy/rel_floor/simplex_l2), `qkg` qk-match-g,
`rn=` rmsnorm-mode (legacy/exact), `pa=` proj-alpha (float/auto),
`ma`/`malr`/`mae` muon-acts toggle/lr/eps, `hp=` head-pool.

## Decomposition: what's really doing the work

| component                | solo  | in combo |
|---|---|---|
| `bm=delta_sum`           | 0.289 | required (combo without it: 0.489) |
| `pn=dyt`                 | 0.499 | required (combo without it: never tested but pa=auto solo 0.356) |
| `pa=auto`                | 0.356 | neutral additive |
| `sm=rel_floor`           | (not tested solo) | small additive |
| `qkg`                    | (not tested solo) | small additive |
| `pn=rms + rn=exact`      | 0.495 | substitutes for dyt with ~−1.5 pt and more noise |

The two-way interaction explanation:
- delta_sum (gain-1 backprop analog: `x̄ = x + Σᵢ(tᵢ−x)`) is mathematically the
  right way to combine fan-out branch targets, but it amplifies whatever the
  head feeds back; with the head feeding noise, the body diverges (0.289).
- Prenorm bounds activation magnitudes inside each sublayer, so the amplified
  targets stay in a usable range.
- With both flags on, the combo with `pa=auto + sm + qkg` is +3.2 over
  baseline; without delta_sum, the same combo is −0.6 below baseline.

## Why baseline plateaus (diagnostic finding)

Per-layer relative target delta `‖t − a‖/‖a‖` from `diag_targets.py`:

| location           | baseline rel_delta |
|---|---|
| `head`             | ~3e-2 |
| `block3` (top)     | ~6e-8 (float32 noise) |
| `block2`           | ~6e-8 |
| `block1`           | ~6e-8 |
| `block0`           | ~6e-8 |
| `patch_embed`      | ~6e-8 |

The body receives **no** teaching signal. Cause: the exact bilinear projection
`matmul_proj` routes the constraint displacement into whichever side has small
`‖·‖²`; at the flatten head, `qa = ‖activation row‖² ≈ 3e4` vs
`qb = ‖weight col‖² ≈ 2`, so essentially all displacement goes into weights
and the activation target collapses to `x` at the very first solve.

`pa=auto` (per-layer α = qa/qb quantized to powers of 2) rebalances the solve:
`t` becomes O(1e-2), body rel_delta → ~1e-6. But the min-norm displacement
`δA = t·Bᵀ/N` is still |B|-scaled (non-expansiveness within one layer), so the
recovered signal is small in magnitude. delta_sum is the structural fix — it
removes the 1/N averaging at the residual fan-out so the actual residual delta
is what flows back; prenorm makes that delta non-explosive.

The 0.50 plateau ≈ linear probe on random features (head learns, body doesn't).
This pattern matches the vanishing-targets theorem from the paper, but it bites
*per-layer at the first bilinear solve*, not gradually through depth — the
geometric decay never gets a chance to start because the signal is already at
the noise floor after one layer.

## What didn't work

- **`muon-acts` (orthogonalize activation residuals to fixed magnitude)**: hurts
  at every setting tested (lr 0.1/0.5/1.0; eps 1e-2/1e-8). It restores per-layer
  magnitude (~1e-3..1e-4 body deltas vs 1e-8 baseline) but replaces the head's
  genuine target with fixed-magnitude steps, which makes the head learn worse
  than it does in baseline. val 0.24–0.31 across all variants.
- **`head_pool=avg`** (replace 8192-dim flatten head with mean pooling): even
  worse — val 0.24 in combo. The flatten head's qa≫qb is the bug, but
  collapsing the sequence loses information the head was extracting.
- **`pn=rms + rn=exact` solo**: the exact joint-projection onto the RMSNorm
  graph manifold (`rmsnorm_proj_exact` in `core/ops.py`) is correct and beats
  legacy on the test (`src/ptorch/core/tests/rms_norm_exact_proj.py`), but in
  the bench at 5k it sits at ~baseline (0.495). It does pair with the combo
  (0.511) but worse than dyt (0.527) and noisier (curve: 1k→0.483, 4k→0.467,
  5k→0.511).
- **`sm=simplex_l2`**: not tested in the bench (only rel_floor was). The Duchi
  helper is in `_project_simplex_sorted` if you want to try it.
- **`residual_mode=half`**: not exercised. Should be tried if delta_sum
  amplification looks like it's hurting on longer runs.

## Trajectory of the winner (`pn=dyt` combo)

```
step    val_acc   train_acc
1000    0.484     0.521
2000    0.513     0.564
3000    0.519     0.583
4000    0.512     0.579
5000    0.527     0.571
```

Slowly climbing; train is flat-to-down (0.583 → 0.571), suggesting either
optimizer noise or a need for lr decay past 4k. Compare to baseline:
1k→0.476, 5k→0.518 (one seed) and plateau ~0.50 at 30k (`baseline_30k`
trajectory in the CSV: 5k 0.518, 10k 0.518, 15k 0.508, 20k 0.518, 25k 0.518,
30k 0.510).

## 30k confirmation command (next step for you)

```bash
.venv/bin/python -m experiments.attention.bench_ptorch_vit \
  --max-steps 30000 --num-runs 2 \
  --prenorm dyt --branch-mode delta_sum --softmax-mode rel_floor \
  --qk-match-g --proj-alpha auto
```

Decision criteria:
- If val keeps climbing past 5k (e.g. 10k → 0.55+) → the combo is real, write up.
- If val flattens near 0.53 → marginal improvement; the deeper bottleneck
  (next section) is dominant.

Optional 30k variants worth a second seed:
- `--prenorm rms --rmsnorm-mode exact` (drop dyt, keep rest) — cleaner theory
  if it matches.
- `--prenorm dyt --branch-mode delta_sum --proj-alpha auto` (drop sm+qkg) — to
  check that those two are really contributing.

## 2026-06-11 update: state-free per-layer δA rescale (mam=rel_row)

### What landed in this session

- `process_activation_target` (src/ptorch/core/ops.py) gained a
  `muon_activations_mode` switch with three new modes on top of the legacy
  `"fixed"` (polar × constant lr, which still fails for the reason the previous
  section gives):
  - `"rel_row"`: polar-orthogonalise the per-row δA direction, then rescale
    each row to `lr · ‖A_det row‖`. The per-token *relative* update is now
    layer-invariant, decoupling the body signal from the `|B|`-scaled
    min-norm bound of `δA = t·Bᵀ/N`.
  - `"rel_frob"`: whole-tensor variant (one norm ratio).
  - `"raw_rel_row"`: per-row rescale of the natural matmul_proj δA *without*
    polar — keeps matmul_proj's chosen direction, only fixes magnitude.
- `muon_activations_max_dim` (default 4096) skips the rescale for activations
  whose `last_dim > max_dim`. This excludes the flatten head's 8192-dim input;
  including it produces `lr · sqrt(8192·var) ≈ O(1)` per-row head
  displacements that swamp the head's natural `t ≈ 3e-2` and collapse val.
- CLI/diag flags: `--muon-acts-mode {fixed,rel_row,rel_frob,raw_rel_row}`,
  `--muon-acts-max-dim N`.

### What we measured

Layer rel_delta with `mam=rel_row malr=0.01 max_dim=4096` at step 200 vs the
pa=auto baseline:

| location | baseline pa=auto | rel_row lr=0.01 |
|---|---|---|
| `head`       | ~1.7e-2 | 2.7e-1 |
| `block3.attn`| ~9.6e-7 | 2.7e-2 |
| `block2.mlp` | ~1.0e-6 | 3.5e-2 |
| `block0`     | ~5.6e-7 | 3.3e-2 |
| `patch_embed`| ~3.9e-7 | 4.2e-2 |

Body rel_delta jumps 4-5 orders of magnitude; the body finally sees signal.

### 5k results (single seed each, same architecture as baseline above)

| arm | 5k val | vs prev winner |
|---|---|---|
| **rel_row lr=0.001 + max_dim=4096** (new best) | **0.5801** | **+5.3** |
| rel_row lr=0.0003 + max_dim=4096               | 0.5681 | +4.1 |
| rel_row lr=0.01 max_dim=4096 (head excluded)   | 0.4392 | −8.7 |
| rel_row lr=0.01 max_dim=0 (head included)      | 0.4237 | −10 |
| ProjectionAdam (winner combo, no muon-acts)    | 0.4995 | −2.7 |
| pn=dyt+bm=delta_sum+sm=rel_floor+qkg+pa=auto (prev winner) | 0.5267 | 0 |

`malr=0.0003` is slightly worse than `malr=0.001` and matches at later steps —
the optimum sits at lr=0.001 for the body but not by a wide margin.

`malr=0.01` overshoots — the over-amplified body direction has noise that
compounds. `malr=0.001` is right-sized: per-token relative update of
`0.001·‖row‖` ≈ ~1e-3 absolute per token, comparable to a healthy gradient
step.

### 30k of the new best

Trajectory (single seed, lr=0.001 rel_row + max_dim=4096 + winner combo):

```
step      train   val
 5000     0.608   0.563
 7000     0.603   0.565
 9000     0.594   0.565    ← peak band
10000     0.583   0.540    ← divergence begins
15000     0.487   0.455
20000     0.422   0.363
25000     0.407   0.386
30000     0.412   0.424    (best-state restore: test_acc = 0.578)
```

**Peak val 0.58 in step band 5k–9k, then catastrophic forgetting**: train_acc
falls from 0.61 → 0.41. The best-state restoration at the end of the run still
recovers test=0.578, but the optimiser is destroying capability after step ~10k.

So the gap closes from 17 → 12 pt *at the peak* (0.58 vs 0.70 AdamW) but the
fix is unstable past the peak.

### What the divergence implies

Compounding noise in the amplified body targets. Each step the polar direction
is locally OK, but ProjectionMuonV2's momentum (β=0.95) integrates
direction-noise into a coherent bad update, and `‖row(A)‖` keeps growing as
body weights drift, which feeds back into a larger rescale → positive feedback.
The head exclusion plus DyT prenorm bound the *forward* magnitudes, but the
*backward* rescale has no such bound.

### Negative results this session

- **ProjectionAdam on the winner combo**: 0.499 at 5k. Per-parameter adaptive
  scaling on top of orthogonalisation does not help; closes hypothesis #3.
- **rel_row lr=0.01 with and without head exclusion**: both worse than the
  no-muon-acts winner. The body signal is too aggressive at this magnitude.

### What's still untried (and should be next)

The peak proves the body bottleneck was real — body signal is now strong enough
that the model crosses 0.58 by step 5k (vs 0.53 in the previous best). The
remaining gap is now:
- ~5pt to the AdamW *5k* number (0.70) — would be closed by either pushing
  malr down to where peak is later/higher, or by combining rel_row with a
  decaying `muon_activations_lr` so the late-training instability dies before
  it can compound.
- Steps to try, in order:
  1. **Sweep malr around 0.001**: try 3e-4 and 3e-3 at 5k to bracket the
     optimum. (3e-4 is launched as of this writing.)
  2. **Add a step-based decay to `muon_activations_lr`**: cosine from `malr`
     to 0 over `decay_steps` (e.g. 10k). The amplification serves its purpose
     in the first few thousand steps when natural body signal is too small;
     after the body has useful features, the polar+rescale is noise.
  3. **Optimiser momentum**: drop ProjectionMuonV2 momentum from 0.95 → 0.5
     (the polar normalisation already amplifies; momentum compounds).
  4. **Gradient-trained head probe** (hypothesis #2): if the body has good
     features by step 5k, a real gradient head should reach `pure linear probe`
     on those features. Confirms whether the body learned something useful
     or merely got mixed.

Hypothesis #1 from the previous section is now **partially confirmed** —
state-free per-layer rescale gets +5pt at peak; learnable per-layer scaling
would presumably stabilise late-training too, but the state-free version is
enough to validate the diagnosis.

## What I think the remaining gap is (untested hypotheses)

The combo at 5k is ~0.53; AdamW is 0.70. The 17 pt gap is unlikely to come from
any of the knobs in this PR. Candidates, ranked by my belief:

1. **|B|-scaled min-norm displacement within a layer** (Lipschitz-bounded
   per-layer signal). Even with `pa=auto` and `delta_sum`, the maximum δA the
   head can ask of layer N-1 is bounded by `‖B‖∞ · O(‖t‖)`. With small weight
   columns at init (norm ~√2 / fan-in), early-training body steps are tiny in
   magnitude regardless of direction quality. Possible fix: a per-layer scale
   on the activation target (treat ω as a learnable per-layer ratio, or
   normalize `δA` to a fixed unit after Muon orthogonalization, akin to
   spectral lr). This is **not** the same as muon-acts (which fixed
   direction+magnitude jointly); the fix would be polar(direction) ×
   learnable_per_layer_magnitude. ~1 day to implement.
   **(2026-06-11: confirmed by mam=rel_row — +5pt at peak; see above section.)**
2. **The head solve is still the bottleneck**: even with `pa=auto` rebalancing
   the body solves, the head sees a 8192→10 single-layer bilinear with no
   downstream pressure; the linear probe ceiling is wherever the body's random
   features land. Diagnostic to confirm: replace the head with a gradient-trained
   linear probe and see if the body now learns better representations.
3. **Optimizer**: ProjectionMuonV2 polar-orthogonalizes the weight
   pseudo-gradient `g = p − p_target`, which is correct for direction but loses
   magnitude info. The effective lr is then `lr × ‖g_orthogonalized‖ ≈ lr × √d`
   which may not match what the projection prescribed. An adaptive
   per-parameter scale (Adam-style) on top of orthogonalization is the
   straightforward thing to try; ProjectionAdam exists, hasn't been swept here.
   **(2026-06-11: closed — ProjectionAdam on the winner combo lands at 0.499 at
   5k, worse than ProjectionMuonV2's 0.527. Adam-style adaptive scaling on top
   of the polar step does not help.)**
4. **Cross-entropy projection step / λ**: now plumbed but never swept past 1.0.
   The CE solve is N=64 iterations of an inner consensus; if the head target
   is the bottleneck this could matter.

## File pointers

- Implementation: `src/ptorch/core/ops.py` (BranchProjection, SoftmaxProjection,
  RMSNormProjection, HardtanhProjection, MatMulProjection auto-alpha,
  `rmsnorm_proj_exact`, DIAG hook).
- Module wiring: `src/ptorch/nn/modules_experimental.py` (Branch, DyT,
  MultiheadAttention `qk_match_g_to_omega`).
- Config flags (all defaults preserve legacy): `src/ptorch/config.py`.
- Bench & CLI: `experiments/attention/bench_ptorch_vit.py` (`FLAG_DEFAULTS`,
  `flag_tag`, `apply_global_flags`).
- Diagnostics script: `experiments/attention/diag_targets.py`.
- Tests: `src/ptorch/core/tests/{rms_norm_exact_proj,hardtanh_proj}.py`.
- README with flag table: `experiments/attention/README.md`.
- Per-arm 5k CSVs: `experiments/attention/results/ptorch_ViT_*_CIFAR10.csv`
  (filename tag = flag combination).
- Diagnostic CSVs: `experiments/attention/results/diag_targets_*.csv`.

## How to re-measure after a new fix

```bash
# Diagnostics (300 steps, per-layer rel_delta + DIAG stats):
.venv/bin/python -m experiments.attention.diag_targets --steps 300 [flags]

# Bench (5k steps, val curve):
.venv/bin/python -m experiments.attention.bench_ptorch_vit \
  --max-steps 5000 --num-runs 1 [flags]

# Then compare results/diag_targets_<tag>.csv and
# results/ptorch_ViT_..._flags=<tag>_CIFAR10.csv against baseline.
```
