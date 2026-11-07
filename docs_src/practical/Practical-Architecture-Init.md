# Practical: architecture and initialisation

**Thesis:** App. B.1.

Two rules from App. B.1, condensed.

## 1. Prefer shallow networks

The vanishing-target theorem (Ch. 4) gives the *why*: every layer's target
map is locally non-expansive, so deep stacks shrink the projection signal
geometrically. Empirically the framework is **more stable and stronger** at
depths 1–4 than at 8+ on MNIST/CIFAR-10 — confirmed by `deep8_config.yaml`
in `experiments/neurips/mlp/`. If your accuracy plateaus, halve the depth before
anything else.

See [[Concepts-Vanishing-Targets]] and [[Experiments-Deep-Diagnostics]] for
the diagnostic figures.

> **STATUS (2026-07-12): partially superseded.** The depth pathology is now
> understood as directional path mis-weighting at residual fan-ins (not raw amplitude
> decay — skips carry amplitude at gain 1) and is largely repaired by merge-ρ fan-in
> equalization: a 1-block GPT trains to 3.2465 vs the gradient twin's 3.1028 with no
> VJP chain. Deeper stacks (L2+) remain gated until L1 parity closes. See the
> "Current understanding" section of [[Concepts-Vanishing-Targets]] and
> `experiments/parameter-golf/STATUS_LEDGER.md`.

## 2. Don't mix `pnn.*` and `nn.*` in the same model

> "Every operation must either leave gradients untouched under standard
> Torch or be replaced by a PTorch equivalent. Mixing in standard Torch
> layers confuses targets with gradients in ways that are easy to miss."
> — *App. B.1*

The recommended path:

- Use `import ptorch.nn.modules as pnn` and stay inside the `pnn.*`
  namespace.
- The override layer (`apply_overrides()`; see [[Reference-Overrides]])
  already covers raw `torch.add`/`torch.sum`/`torch.mul`/`torch.square`
  inside your model, so you do not need projection-aware wrappers for those.

The **hybrid-architecture** experiment
(`experiments/cnn_hybrid/bench_ptorch_hybrid_arch.py`) deliberately
violates this rule to measure what happens; see [[Experiments-CNN]] for the
findings.

## 3. Initialisation

App. B.1 Fig. B.1 sweeps three init schemes (normal `std=0.01`, He-normal,
scaled orthogonal) at depths 2, 4, 8. Key observations:

- At depth 2, all three schemes work; convergence speed roughly matches the
  gradient-trained baseline.
- At depth 4+, **He-normal** and **scaled orthogonal** clearly dominate the
  small-`std` normal init.
- PTorch currently reuses the gradient-training init heuristics — there is
  **no projection-specific init research** yet. The thesis flags this as a
  high-value open question.

`pnn.Linear` defaults to `kaiming_normal_` (`nn.init.kaiming_normal_(self.weight)`
in `src/ptorch/nn/modules.py:46`), i.e. He-normal. `pnn.Conv2D` inherits
through its inner `Linear`.

## See also

- [[Concepts-Vanishing-Targets]] — why shallow.
- [[Reference-Overrides]] — why mixing is safe at the op level but not at
  the module level.
- [[Practical-Optimizer-Loss]] — once architecture is settled, tune the
  optimizer + proximal-loss knobs.
