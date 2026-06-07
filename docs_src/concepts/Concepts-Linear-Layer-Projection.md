# Linear-layer projection

**Thesis:** §3.1 — fused consensus + bilinear constraint; choice of geometry.
**Code:** `matmul_proj`, `matmul_proj_linf`, `MatMulProjection`,
`MatMulProjectionLinf` in `src/ptorch/core/ops.py` (lines 90–411).

## What it does

The linear layer's constraint set is

    A = { (A, B, Z) | A B = Z }

with `A ∈ ℝ^{M×K}` activations, `B ∈ ℝ^{K×N}` weights, `Z ∈ ℝ^{M×N}` output.
Three projection design choices distinguish PTorch from PJAX:

1. **Consensus and bilinearity are fused** (§3.1.1). PJAX duplicates `B`
   per datapoint and projects each pair `(Aᵢ, B_i, Zᵢ)` independently, then
   averages `B_i`. PTorch projects directly onto the joint constraint
   `A B = Z` for the full batch in a single shot — no parameter replication.
2. **Two geometries**, selected by the `norm=` argument to `pnn.Linear`:
   - `l2` → `matmul_proj` solves the L₂-fused problem via a 1-step damped
     Newton iteration on a 1-D root equation (the *bracketed Newton* of
     Elser, *Learning Without Loss*, eq. 52; see comments in
     `src/ptorch/core/ops.py:286-296`).
   - `linf` → `matmul_proj_linf` solves the Chebyshev-norm variant via the
     piecewise-linear Newton scheme of §3.1.2.
3. **Caching of warm-start scalars** between iterations — the `t` (L₂) or
   `eps` (L∞) Lagrange/radius variable is stored in `proj_cache` and reused
   across consecutive backward calls (`MatMulProjection.forward` ctx; see
   `src/ptorch/nn/modules.py:36-78`).

## How to use

```python
import ptorch.nn.modules as pnn

# L2 fused projection (default)
fc = pnn.Linear(784, 256)

# L∞ (Chebyshev) projection — preferred when activations are bounded /
# binary; selected per-call too.
fc = pnn.Linear(784, 256, norm='linf')
```

Bias is folded into the projection by appending a constant `1` column to
the activation and the bias row to the weight matrix
(`src/ptorch/nn/modules.py:62-65`). The geometry of the projection therefore
treats `(weight, bias)` uniformly.

## Choosing the geometry (§3.1.3)

The thesis argues that **L∞** matches the natural scale of ReLU/Linear+ReLU
stacks (where activations are non-negative and bounded layerwise) and that
**L₂** is better suited to unconstrained pre-activations. The
`norm_comparison` benchmark in `experiments/mlp/` produces the figure that
backs this argument; see [[Experiments-MLP]].

## How it maps to the thesis

| Thesis equation | Code line | Meaning |
|---|---|---|
| §3.1.1 fused KKT (Eq. 3.7 in the PDF) | `src/ptorch/core/ops.py:301-326` | One Newton step on the rooted-quadratic surrogate |
| §3.1.1 closed-form reconstruction | `ops.py:328-343` | `A_proj`, `B_proj`, `Z_proj` from the converged `t` |
| §3.1.2 L∞ piecewise problem | `ops.py:140-196` | Newton on the diamond expansion |
| §3.1 bias absorption | `nn/modules.py:62-65` | Append-1 trick |

## See also

- [[Reference-Ops]] — signatures of every `*_proj` helper.
- [[Reference-NN-Modules]] — how `Linear` dispatches between the two.
- [[Concepts-Cyclic-Projections]] — where this projection sits in the cycle.
