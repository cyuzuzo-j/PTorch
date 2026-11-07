# Reference: `config.py`

> **STATUS (2026-07-12): a campaign behind.** `config.py` is now ~385 LOC (was 57 when
> this page was written); the 2026-07 GPT campaign added many keys (hopfield attention,
> softmax/rmsnorm modes, transit-VJP gates, merge-ρ, TD scope, …), several registered at
> import time by extension modules rather than in `config.py` itself. For which knob
> settings work and which are falsified, see key docstrings in `src/ptorch/config.py`
> and `experiments/parameter-golf/STATUS_LEDGER.md`.

**File:** `src/ptorch/config.py` (57 LOC)
**Code:** `from ptorch.config import config`

A process-wide `Config` dataclass that the ops and modules read at runtime.
Mutations go through `config.update(key, value)` (validates against the
field schema) and a `with config.projections(enabled=False):` temporary
override exists for parity tests.

## Public fields

| Field | Default | Read by | Effect |
|---|---|---|---|
| `use_projections` | `True` | every module + override | If `False`, modules fall back to standard PyTorch and the override dispatchers route to the original ops. |
| `projection_alpha` | `1.0` | `pnn.Linear`, `pnn.Conv2D` | Scales the per-layer `alpha` knob (weight–data balance in the L₂ projection). |
| `projection_g` | `1.0` | `pnn.Linear`, `pnn.Conv2D` | Scales the per-layer `g` (output-target weight). |
| `cross_entropy_num_steps` | `5` | `CrossEntropyProjection` | Inner fixed-point iterations for the proximal CE backward. |
| `cross_entropy_lambda` | `5.0` | `CrossEntropyProjection` | Proximal strength `λ` (smaller = softer loss). |

> **STATUS (2026-07-12):** GPT-campaign verdict on the CE knobs (`CE_STEPS` in the
> trainer): **5 steps confirmed keeper off-VJP** — the ~5× hot prox seed is co-adapted
> to heuristic transport; `CE_STEPS=1` (exact ∇CE seed) helps **only** with a transitive
> VJP chain that conserves it (−0.13 with, **+0.25 falsified without**). See
> `experiments/parameter-golf/STATUS_LEDGER.md`.
| `use_muon_activations` | `False` | `process_activation_target` | Orthogonalise the activation-target residual with Polar-Express. |
| `muon_activations_lr` | `1.0` | `process_activation_target` | Step size for the orthogonalised activation update. |

## API

### `config.update(key, value)` *(config.py:31-37)*
Atomic set under an internal lock; raises on unknown keys.

### `config.reset()` *(config.py:39-43)*
Restore all fields to their dataclass defaults.

### `config.snapshot() -> dict` *(config.py:45-50)*
Snapshot of every public field — convenient for logging.

## YAML wiring

The benchmark scripts read these knobs from
`experiments/*/config.yaml`. For example `experiments/neurips/mlp/config.yaml` sets:

```yaml
use_muon_activations: false   # → config.use_muon_activations
norms: [linf]                 # → pnn.Linear(..., norm='linf') per construction
```

See each experiment page for the full per-YAML knob list:
[[Experiments-MLP]], [[Experiments-CNN]], [[Experiments-Attention]].

## See also

- [[Reference-Ops]] — `process_activation_target` reads `use_muon_activations`.
- [[Reference-NN-Modules]] — `Linear` reads `projection_alpha`/`projection_g`.
- [[Practical-Optimizer-Loss]] — sensible values for the CE-prox knobs.
