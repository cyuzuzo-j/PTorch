# PTorch Wiki

**PTorch** trains neural networks by **cyclic projections** instead of
backpropagation. Each layer's backward pass returns the nearest point that
satisfies the layer's local constraint — a *projection target* — and the
optimizer treats `p − p_proj` as a pseudo-gradient.

This wiki is a **code-first reference**: every page anchors on a concrete
function in `src/ptorch/` or a script in `experiments/`, then points back to
the relevant chapter/equation of the thesis it implements.

## Thesis

> Cyuzuzo Jambé, J. *PTorch: Narrowing the Gap Between Projection and
> Gradient-Based Learning.* M.Sc. thesis, KU Leuven, 2025–26.
> Supervisor: Prof. dr. ir. Panos Patrinos. Assistant: Ir. Jan Quan.

The PDF lives at `thesis__2_ (20).pdf` in the repo root.

## Map

**Start here**
- [[Getting-Started]] — install, quick start, smoke tests
- [[Glossary]] — feasibility problem, projection, target, proximal, …

**Concepts** (light theory, anchored to code)
- [[Concepts-Feasibility-Framing]] — *thesis Ch.2*
- [[Concepts-Cyclic-Projections]] — *thesis §3.2*
- [[Concepts-Linear-Layer-Projection]] — *thesis §3.1*
- [[Concepts-Vanishing-Targets]] — *thesis Ch.4*

**API reference**
- [[Reference-Ops]] — `src/ptorch/core/ops.py`
- [[Reference-NN-Modules]] — `src/ptorch/nn/modules.py`
- [[Reference-Optimizers]] — `src/ptorch/optim_static.py`
- [[Reference-Overrides]] — `src/ptorch/core/overrides.py`
- [[Reference-Config]] — `src/ptorch/config.py`

**Experiments**
- [[Experiments-MLP]] — *thesis §5.1*
- [[Experiments-Deep-Diagnostics]] — *thesis Ch.4, §5.2*
- [[Experiments-CNN]] — *thesis §5.3, App. A*
- [[Experiments-Attention]] — *thesis §5.4*
- [[Experiments-Non-Differentiable]] — *thesis App. A.3*

**Practical guidance** (from the thesis appendices)
- [[Practical-Architecture-Init]] — *thesis App. B.1*
- [[Practical-Optimizer-Loss]] — *thesis App. B.2, B.4*
- [[Useful-Projections]] — *thesis App. D*
