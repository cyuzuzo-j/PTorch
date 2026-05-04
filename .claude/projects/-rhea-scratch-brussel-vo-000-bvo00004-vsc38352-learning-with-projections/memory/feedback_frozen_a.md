---
name: FrozenA weight targeting is required
description: User confirmed frozen_a_weights=True is needed — gives 50% vs 20% without it
type: feedback
---

Use FrozenA (`frozen_a_weights=True`) for weight updates. It is required for the ResNet benchmark to work.

**Why:** Empirical results show frozen_a_weights=True gives ~50% accuracy on CIFAR-10, while frozen_a_weights=False gives only ~20% (near random). The joint bilinear projection (matmul_proj) averages weight targets over M=batch*H*W positions, making weight signal vanishingly small without FrozenA. The K*K least-squares solve gives meaningful weight updates.

**How to apply:** Always set `frozen_a_weights=True` in bench_ptorch_resnet.py. Previous singular matrix errors were with ResNet-18 (wider layers) — ResNet-8 works fine. User ran both configs and confirmed this.
