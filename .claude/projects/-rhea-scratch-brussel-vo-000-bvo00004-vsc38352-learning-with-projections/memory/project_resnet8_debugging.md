---
name: ResNet-8 CIFAR-10 accuracy debugging
description: Ongoing effort to push ptorch target-propagation ResNet-8 from ~39% to 95% on CIFAR-10. Tracks root causes found, fixes applied, and next steps.
type: project
---

## Goal
Get ResNet-8 (`experiments/cnn_benchmarks/bench_ptorch_resnet.py`) to 95% accuracy on CIFAR-10 using the ptorch projection-based target propagation framework.

## Best result so far
39.4% val accuracy (diag_resnet19) with: frozen_a_weights=True, muon_activations_lr=0.5, norm_preserve=False, ProjectionSGD lr=0.03, batch_size=256. But FrozenA is now rejected (see below).

## Key bugs fixed (sessions prior to 2026-05-03)

1. **ResidualAdd.backward had swapped arguments** (bench_ptorch_resnet.py:70-72): The first arg to `process_activation_target` must be the CURRENT tensor value. Was receiving the OTHER tensor. Fixing this went from ~10% to ~37%.

2. **Tiny weight updates from joint projection**: In `matmul_proj` (ops.py ~line 446), `B_eff_proj = (1/M) * (...)` averages over M = batch*H*W (e.g. 16384), making weight corrections vanishingly small. Was temporarily fixed with FrozenA least-squares solve, but that's now rejected.

3. **Feature shrinkage from Muon activation targeting**: `process_activation_target` (ops.py:66-94) computes `A_new = A - lr * g_muon` where g_muon has unit spectral norm. Fixed step size of lr=0.5 systematically reduces feature norms over training. Features shrink from ~0.74 to ~0.31 over 10K steps.

## Architecture (current)
```
ResNet8_PTorch:
  block1: BasicBlock(3 -> 128, stride=2)   # 32x32 -> 16x16
  block2: BasicBlock(128 -> 512, stride=2)  # 16x16 -> 8x8
  block3: BasicBlock(512 -> 1024, stride=2) # 8x8 -> 4x4
  ProjectedGlobalAvgPool                     # 4x4 -> 1x1
  head: Linear(1024 -> 10, norm="inf")
  loss: CrossEntropy
```
Each BasicBlock: conv1(3x3) -> LeakyReLU(0.1) -> conv2(3x3) + shortcut(1x1) via ResidualAdd -> LeakyReLU. Conv2 weights initialized to zero (identity residual).

## User decisions (2026-05-03)
- **FrozenA is rejected**: User explicitly said "I still think the FrozenA is a bad idea and should be avoided." Reasons: (1) K*K linear system solve is O(K^3) and expensive for large K; (2) singular matrix errors with wider networks (confirmed: FrozenA crashed at step ~1500 in diag_resnet22 C1); (3) freezing activations limits joint optimization.
- **ResNet-18 removed**: User added it to test, got singular matrix error, asked to remove it. Deleted from bench_ptorch_resnet.py on 2026-05-03.
- **Benchmark updated**: frozen_a_weights=False in bench_ptorch_resnet.py.

## Remaining bottlenecks (as of 2026-05-03)

### 1. Weight updates too small without FrozenA
The joint bilinear projection (matmul_proj) averages weight targets over M spatial positions. With M=16384, weight signal is ~0. Need to amplify it.

**Proposed fix being tested (diag_resnet22)**: Increase `projection_g` from 1.0 to 5.0. In the cost function `||A_new-A||^2 + alpha*||B_new-B||^2 + g*||Z_new-Z||^2`, higher g makes `target_penalty = omega^2/g^2 = 1/25` (vs 1.0), forcing the Newton solver to find larger t values -> larger weight AND activation changes.

### 2. Feature shrinkage from fixed-step Muon
`process_activation_target` subtracts `lr * g_muon` (unit spectral norm) regardless of how close features are to targets. Over time features shrink.

**Proposed fix**: `muon_activations_scale=True` (already implemented in ops.py:83-85). Makes step proportional to `||A - A_proj||` — adaptive, shrinks as features approach targets.

### 3. Conv branch targets discarded (detach pattern)
In BasicBlock.forward: `conv_in = x.detach().requires_grad_(True)` means only the skip connection propagates targets between blocks. The conv branch (where most capacity lives) contributes NOTHING to earlier layers' targets.

**Proposed fix**: Replace detach with `AverageGradient.apply(x, 2)` so both skip and conv branches propagate targets, averaged at the split point. Implemented as `BasicBlockAvgGrad` in diag_resnet22.

### 4. No data augmentation
InfiniteCifarLoader supports `aug={'flip': True, 'translate': 4}` but it's not enabled. Standard CIFAR-10 augmentation could help significantly.

### 5. Only 10K training steps
Standard CIFAR-10 training uses ~39K steps (200 epochs). We may simply need more training time.

## Active diagnostic: diag_resnet22.py (started 2026-05-03 ~15:25)

Tests 6 configs (7500 steps each, no FrozenA):

| Config | g | Muon W | Muon W scale | Muon A scale | Block | Aug | opt_lr |
|--------|---|--------|-------------|-------------|-------|-----|--------|
| C1: baseline g=1 | 1.0 | off | - | off | detach | no | 0.03 |
| C2: g=5 | 5.0 | off | - | off | detach | no | 0.03 |
| C3: g=5+muon_w | 5.0 | on | True | off | detach | no | 0.1 |
| C4: +muon_a_scale | 5.0 | on | True | True | detach | no | 0.1 |
| C5: +avg_grad | 5.0 | on | True | True | AvgGrad | no | 0.1 |
| C6: +augmentation | 5.0 | on | True | True | AvgGrad | yes | 0.1 |

Status: **Running** (output buffered by Python, not yet flushed). Process PID was 3848543.

Tracks `w_change` = relative pseudo-gradient norm (measured between backward() and optimizer.step()) to verify whether higher g actually amplifies weight signal.

## Key code locations

- **Benchmark**: `experiments/cnn_benchmarks/bench_ptorch_resnet.py`
- **Config**: `experiments/cnn_benchmarks/config.yaml` (ptorch_optimizer: ProjectionSGD, lr: 0.03)
- **Framework config**: `frameworks/ptorch/config.py` (defaults dict, Config singleton)
- **Core projection ops**: `frameworks/ptorch/core/ops.py`
  - `process_activation_target` (line 66): Muon orthogonalization for activations
  - `process_weight_target` (line 98): Muon for weights (disabled by default)
  - `compute_weight_target_frozen_a` (line 128): FrozenA K*K solve (DO NOT USE)
  - `matmul_proj` (line 368): Joint bilinear projection via Newton method
  - `MatMulProjection.backward` (line 482): Main backward pass, dispatches to frozen_a or joint
  - `AverageGradient` (line 154): Averages backward targets from multiple paths
- **Optimizers**: `frameworks/ptorch/optim_static.py` (ProjectionSGD, AlternatingProjections, etc.)
- **Conv2D**: `frameworks/ptorch/nn/modules.py:503` — unfolds to patches, uses Linear for projection
- **Data**: `experiments/shared/data.py:651` (InfiniteCifarDataModule), line 521 (InfiniteCifarLoader with aug support)

## How the projection backward works (simplified)
1. CrossEntropy gives target for logits
2. Head Linear: MatMulProjection computes A_proj (activation target) and B_proj (weight target)
3. ProjectedGlobalAvgPool: distributes target across spatial positions
4. Each BasicBlock's ResidualAdd: splits target into skip_target and out_target
5. Skip path: shortcut conv -> MatMulProjection -> target for previous block
6. Conv path: conv2 -> relu -> conv1 -> target stops at detach (or averages if using AvgGrad)
7. Optimizer (ProjectionSGD): converts p_proj to pseudo-gradient (p - p_proj), applies SGD with lr

## Diagnostic history (for reference)
- diag_resnet10-12: Early conv architecture debugging
- diag_resnet13-14: FrozenA development, AlternatingProjections vs ProjectionSGD
- diag_resnet15: 3-block model failure (before ResidualAdd bug fix)
- diag_resnet16-17: Feature shrinkage investigation
- diag_resnet18-19: Best configs found (39.4%)
- diag_resnet20-21: norm_preserve testing (results never read — likely obsolete now)
- diag_resnet22: Current — testing non-FrozenA structural fixes

## Environment
- Cluster: VSC (Flemish Supercomputer), RHEA scratch filesystem
- Python venv: `.venv/` in project root (activate with `source .venv/bin/activate`)
- PyTorch 2.10.0+cu128, CUDA available
- Wandb project: "pjax"
- Note: Python stdout is fully buffered when piped to file — use `python -u` or `PYTHONUNBUFFERED=1` to see output in real time
