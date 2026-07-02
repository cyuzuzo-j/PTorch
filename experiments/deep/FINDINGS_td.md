# TD(λ)-style traces vs. vanishing targets: what actually breaks, and the direction pipe

Testbed: identity recovery (Section 5.2) — linear MLPs, width d=32, batch 16, MSE
projection loss, `ProjectionMuonV2`. All numbers measured in this repo
(`experiments/deep/td_lambda_identity.py`, diagnostics in this file's history).
`config.td_lambda` gates everything; λ=0 is bitwise-identical to the pre-trace code
(verified by `test_td_lambda_equiv.py`, SHA-256 over all weights after 20 steps).

## 1. The local solve is adjoint transport with a lossy amplitude budget

Linearizing the bilinear constraint (A+δA)(B+δB) = AB − s under
min ‖δA‖² + α‖δB‖² gives δA ∝ s·Bᵀ·M: the activation share of the residual is
transported through **Bᵀ — backprop's adjoint** — by the projection itself. This is
visible in `matmul_proj`'s reconstruction (`t_inv_denom @ Bᵀ` term). Measured at α=1,
small residual: **cos(δA, s@Bᵀ) = 0.95**.

The *amplitude* split between activations and weights is
τ ≈ α·q_b / (α·q_b + q_a), with q_a = mean row ‖A‖² (~33 at init) and q_b = mean col
‖B‖² (~2.1) ⇒ τ ≈ 0.06 per layer — matching the ~order-of-magnitude-per-layer
collapse of Fig 5.3a. After ~5–6 hops the amplitude reaches solver/dtype noise
(~1e-7) and the direction is annihilated with it (measured: cos = 0.07 at
‖r‖ = 2e-8).

> Vanishing targets = correctly-directed amplitude decaying geometrically until
> numerical annihilation. The information is intact until underflow; the problem is
> numerics, not routing.

## 2. Variant A (vector trace) restores magnitude but transports zero information

Correct target at depth k is Jᵀs (J = downstream weight product). The trace delivers
s verbatim — identity transport, valid only when J ≈ I (residual nets). For random J,
E|cos(s, Jᵀs)| ≈ 1/√d = 0.177; measured per-layer alignment 0.05–0.24 (i.e. cos² ≈
1/d ⇒ ~3% signal, 97% isotropic noise). Consequences, measured over the full sweep
(5000 steps, 3 seeds): step-1 profile flattens exactly as λ dictates (decay rate
≈ max(λ, c)), but final MSE is flat at L=8, modest at L=4 (λ=0.3: 4.5e-3 vs 6.4e-3),
and 16× *worse* at L=2 (λ-independent ~2e-3 noise floor from the persistent
misaligned component). Kept in the code (`td_mode="vector"`) as the negative control.

## 3. K inner passes cannot rescue a misaligned signal

Per pass, the useful component reaching depth k is c^k (TD(0)) or a fixed ~1/√d
fraction (trace). K passes accumulate at most K× — linear — against an exponential
deficit (rescue needs K ≈ c^-k ≈ 10^k). For the trace, the junk component comes from
the same seed and accumulates coherently at the same rate as the signal: SNR ≈ 1/√d
independent of K. Repetition multiplies magnitude, never information.

## 4. Neither raising α nor naive re-amplification works: the saturation regime

Two more measured dead ends, same root cause:

- **α sweep** (unit residual): amplitude delivered to activations grows with α
  (0.001 → 1.1 of the residual) but cos collapses 0.95 → ~0.1–0.28 past α ≈ q_a/q_b.
- **Uncapped Variant B** (scalar floor, amplify to ‖seed‖): magnitude profile perfect
  (λ=0.5 halves per hop; λ=1 flat at the seed norm), but alignment decays
  0.83 → 0.52 → 0.15 → ~−0.1 and the delivered residual becomes **91–99.6%
  row-parallel to the activation itself** at depth.

Mechanism: the solve is adjoint-faithful only for residuals small relative to the
activations. Measured fidelity window (cos(δA, s@Bᵀ) vs relative amplitude ‖s‖/‖Z‖):

| rel. amplitude | 1e-6 | 1e-4 | 1e-3 | 1e-2 | 1e-1 | 3e-1 | 1.0 |
|---|---|---|---|---|---|---|---|
| cos | 0.24 | 0.94 | 0.95 | 0.95 | 0.95 | 0.91 | 0.49 |

Below ~1e-5: numerical noise. Above ~0.3: the Newton parameter t saturates and the
activation move degenerates to the A-parallel rescaling term (t²/(α−t²))·A — an
artifact, not transport. Any scheme that pumps amplitude back to seed scale pushes
every subsequent hop into saturation. Deflating the known A-parallel component
recovers ~1 hop only (measured): the transported part itself has already been through
saturated solves.

Related hazard found on the way: a **stale t cache** from a saturated solve poisons
the next 1-step Newton solve entirely (cos 0.31, |δA| pinned at the stale scale
regardless of the actual residual). Alignment measurements must snapshot/restore
`proj_cache` (done in `measure_alignment`).

## 5. The fix: a direction pipe — seed-coupled floor capped into the fidelity window

Under Muon, weight-update *magnitude* is free (polar normalization); the backward
chain only needs to carry *direction*, at any amplitude inside the fidelity window.
Variant B (`td_mode="norm"`) with the linear-regime cap (`td_eps_lin`, default 1e-2):

```
floor ← λ·floor + (1−λ)·‖r_local‖          # scalar trace, seeded at ‖seed‖
amp   ← min(floor, td_eps_lin · ‖A‖)       # stay inside the fidelity window
write   A_det − max(1, amp/‖r_local‖) · r_local
```

Only a scalar crosses layers ⇒ frame-invariant, width-agnostic (no dimension bridge
needed), λ=0 is an exact no-op, and the floor anneals to zero with the loss (unlike
the ‖A‖-coupled `use_muon_activations` rescue, which never anneals). Measured step-1
alignment at depth 8 (cold caches, fresh model):

| | h1 | h2 | h3 | h4 | h5 | h6 | h7 | h8 |
|---|---|---|---|---|---|---|---|---|
| λ=0 (baseline) | 0 | 0 | 0 | 0 | 0 | +0.07 | +0.78 | +1.00 |
| norm + cap, λ≥0.5 | **+0.17** | +0.18 | +0.20 | +0.25 | +0.26 | +0.42 | +0.78 | +1.00 |

+0.17 at h1 vs ~0.04 for a random 512-dim vector: genuine transported signal at a
depth where the baseline delivers literally zero. (λ values coincide at step 1
because the cap binds while the seed is large; they differentiate as the floor
anneals below the cap.)

## 6. Why final MSE rises with depth — and why frozen-top "hardening" is wrong

An earlier version of this analysis claimed the plain testbed cannot see propagation
quality because "the top layer alone can solve linear identity" (any invertible map
is one layer's reach), and proposed freezing the top K layers to force early-layer
learning. **Refuted** (credit: user objection — if top-layer-suffices were true,
final MSE would be flat in depth; it rises 1.2e-4 → 2.2e-2 from L=2 to L=8):

- The map the top layer would need is (W_{L-1}···W_1)⁻¹ — the inverse of a random
  product. Measured: a 7-fold kaiming 32×32 product has κ ≈ 5e8, ‖M⁻¹‖ ≈ 1e7.
  Reachable in principle, unreachable under bounded-norm updates in any finite budget.
- Measured with *true gradients* (Adam, 2000 steps, plain task): top-layer-only
  training stalls at MSE 0.96 at L=8 (≈ predicting zero) vs 0.30 for all layers.
  Early-layer participation is genuinely required — the practical solution is a
  well-conditioned factorization with every layer contributing an O(1) factor.

Consequences: (a) the plain testbed's MSE-vs-depth rise **is** a propagation metric —
no hardening needed; (b) `--freeze-top` is worse than unnecessary: it makes the
required bottom-stack solution the inverse of a κ~1e5+ frozen random product, which
is unreachable for *every* method — all methods stall together and nothing is
discriminated. The flag remains in `td_lambda_identity.py` but should not be used
for headline comparisons. The honest reference for loss-vs-depth plots is instead an
all-layers true-gradient (Adam) run on the same task and budget.

## 7. Sweep results (5000 steps, 3 seeds, plain testbed, loss/16 convention)

Final MSE, norm mode (`td_eps_lin = 1e-2`), against the λ=0 baseline and an
all-layers true-gradient Adam reference (lr 1e-3, same task/budget):

| λ_TD | L=2 | L=4 | L=8 |
|---|---|---|---|
| 0.0 (baseline) | 1.2e-4 | 6.4e-3 | 2.22e-2 |
| 0.3 | 1.7e-4 | **8.6e-5** | **1.17e-2** |
| 0.5 | 1.7e-4 | 8.9e-5 | 1.23e-2 |
| 0.7 | 1.7e-4 | 1.1e-4 | 1.67e-2 |
| 0.9 | 1.7e-4 | 1.3e-4 | 2.85e-2 |
| 1.0 | 1.7e-4 | 1.1e-4 | 3.57e-2 |
| Adam (backprop ceiling) | 3.8e-14 | 1.1e-3 | 1.14e-2 |

Headlines:
- **L=4: 75× better than baseline** (8.6e-5 vs 6.4e-3), beating even the Adam
  reference mean (bimodal: two seeds ~1e-6, one 3e-3).
- **L=8: 1.9× better than baseline and lands on the backprop ceiling**
  (1.17e-2 vs Adam's 1.14e-2); per-seed ranges of λ=0.3 (1.07–1.38e-2) and baseline
  (2.02–2.44e-2) do not overlap.
- Moderate λ ≈ 0.3–0.5 is optimal, exactly as the calibration predicted; λ ≥ 0.9 is
  *worse* than baseline at L=8 (3.6e-2 at λ=1): a floor that never anneals keeps
  injecting amplified noise near convergence.
- L=2 pays a small dilution tax (1.7e-4 vs 1.2e-4).
- Alignment evolution (mean cos over the deep half h1–h4 at L=8): +0.31 at step 1
  for all λ (the cap binds), annealing to ~+0.05 by step 5000 at λ=0.3–0.5, and to
  ~0/negative at λ=1 — the annealing floor is what separates good λ from bad.

Bottom line: on this testbed, the direction pipe (Variant B + linear-regime cap)
removes the vanishing-target penalty up to L=8, closing the gap to backprop at equal
budget. The binding constraint was never magnitude and never credit assignment — it
was keeping a correctly-transported direction inside the solver's fidelity window.

## 8. Nonlinear MLPs (MNIST, `experiments/mlp/deep_mlp_config.yaml`)

ReLU MLPs (784→512^k→10), CrossEntropy projection loss (now seeds the trace),
ProjectionMuonV2, batch 32, 5000 steps, 3 runs; 2×2 over the existing
muon-activations rescue × the trace (norm mode, λ_TD = 0.3). Test accuracy:

| arm | MNIST_1L | MNIST_4L |
|---|---|---|
| muon rescue on, trace off (config baseline) | 95.80 | 93.63 |
| muon rescue on, trace on | 95.78 | 93.75 |
| muon rescue off, trace off | 95.88 | 92.14 |
| **muon rescue off, trace on** | **97.44** | **94.11** |
| torch Adam reference (legacy, 10 runs) | 97.76 | 97.08 |

- The pure trace (muon off, td on) is the best projection configuration on both
  tasks: **+1.6 pts at 1L, +2.0 pts at 4L** over its own baseline, run ranges
  non-overlapping, and it beats the muon-activations rescue everywhere.
- The trace **supersedes** `use_muon_activations`: with the fixed-magnitude muon
  rescale active, the trace adds ~nothing (both act on activation-target magnitude;
  muon's non-annealing fixed step also inflates `r_local` past the floor so the
  `max(1,·)` rarely engages) — and the muon arm itself is dominated. Recommended
  setting going forward: `use_muon_activations=False`, `td_mode="norm"`,
  `td_lambda≈0.3`.
- vs backprop: 1L nearly closed (97.4 vs 97.8); at 4L a ~3 pt gap remains — the
  nonlinear deep case is improved but not closed (ReLU/softmax nodes still consume
  signal between the re-amplifying Linear solves; per-node floor participation is
  the natural next step).

### 8b. Repeater on non-linear nodes (v2)

`td_repeat_target` / `process_node_target` (`core/ops.py`): every non-bilinear
projection backward (ReLU, LeakyReLU, Hardtanh, Softmax, steps/quantized, pools,
ConvPatch, MaskedAdd, RMSNorm — 13 sites) rescales its input-target residual back up
to the current floor, capped at `td_eps_lin·‖x‖`, **without updating the floor** —
λ-blending stays exclusive to the bilinear solves, so activation nodes are amplitude
repeaters, not extra trace hops. `MatMulProjectionLinf` and `BranchProjection`
deliberately untouched. λ=0 stays bitwise-identical (test-verified). Results
(same protocol as §8; v1 archived in `results/deep_mlp/td_v1_linear_only/`):

| muon rescue, trace on | MNIST_1L v1 → v2 | MNIST_4L v1 → v2 |
|---|---|---|
| off | 97.44 → 97.45 | 94.11 → **94.48** |
| on  | 95.78 → 95.77 | 93.75 → 93.81 |

+0.37 pts at 4L (where four ReLUs sit between the solves), no effect at 1L (one
ReLU, nothing to heal) — matching the mechanism. Cumulative at 4L, muon off:
92.14 (trace off) → 94.11 (Linear-only trace) → 94.48 (with repeater); Adam 97.08.

### 8c. Hyperparameter probes are flat — the residual gap is not a tuning problem

MNIST_4L, muon off, 3 runs each: λ=0.5 → 94.40, λ=0.7 → 94.47, λ=0.3 annealed
linearly to 0 → 94.49, cap 3e-2 → 94.23, vs the λ=0.3/cap-1e-2 reference 94.48.
The trace is insensitive to λ over 0.3–0.7, annealing buys nothing (late-training
trace noise is not the binding constraint at this budget), and a wider cap slightly
hurts (consistent with the fidelity-window measurement). Note also that even Adam's
4L (97.08) trails its own 1L (97.76) on MNIST at this budget — depth does not pay on
MNIST for any method; the meaningful target is projection-4L ≈ Adam-4L. Remaining
suspects for that ~2.6 pt gap: per-hop transport fidelity (num_iters, artifact
deflation — `td_deflate`), per-row floors, and momentum-based noise averaging.

### 8d. Deflation closes the gap; more Newton steps are actively harmful

`td_deflate` (config-gated, off by default): before the floor rescale, remove from
`r_local` its per-row component parallel to the activation — the bilinear solve's
known self-rescaling artifact, which is not transported signal. MNIST_4L, muon off,
λ=0.3, 3 runs:

| variant | test acc |
|---|---|
| trace, num_iters=1 (8b reference) | 94.48 |
| **trace + deflate** | **97.21** [97.12, 97.12, 97.37] |
| trace, num_iters=3 | 84.73 |
| trace + deflate, num_iters=3 | 15.94 (diverged) |
| no trace, num_iters=3 | 86.51 |
| no trace, num_iters=1 (baseline) | 92.14 |
| Adam reference | 97.08 |

- **Trace + deflation reaches 97.21 — at/above the Adam-4L reference (97.08).** The
  activation-parallel artifact was the dominant remaining noise: deflating it is
  worth +2.7 pts, more than everything since the original trace combined.
- **num_iters=1 is load-bearing damping, not a compromise**: 3 Newton steps per
  solve collapse accuracy by ~6-10 pts even without the trace (larger |t| ⇒ more
  aggressive exact projections ⇒ bigger artifact + overshoot), and combined with
  deflation it diverges (deflated r shrinks ⇒ the floor rescale amplifies harder ⇒
  out of the fidelity window). Do not raise num_iters in this regime.
- Recommended recipe: `use_muon_activations=False`, `td_mode="norm"`,
  `td_lambda=0.3`, `td_eps_lin=1e-2`, `td_deflate=True`, `num_iters=1`.
- With deflation, MNIST_1L reaches 97.60 (vs Adam 97.76). Final scoreboard:
  projection 1L/4L = 97.60/97.21 vs Adam 97.76/97.08 — the projection framework's
  depth tax (0.39 pts) is now *smaller* than backprop's (0.68 pts) on this task,
  and 4L is at parity with backprop.
