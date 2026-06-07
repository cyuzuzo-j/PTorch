# Reference: `config.py`

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
`experiments/*/config.yaml`. For example `experiments/mlp/config.yaml` sets:

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
