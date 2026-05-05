import threading
import contextlib
from dataclasses import dataclass, field, fields, MISSING
from typing import Optional


@dataclass
class Config:
    """Projection framework configuration.

    Read settings as plain attributes:  config.use_projections
    Mutate with validation:             config.update("muon_weights", True)
    Temporary override:                 with config.projections(enabled=False): ...
    """

    # ---- cross-entropy constraint -------------------------------------------
    cross_entropy_num_steps: int = 5
    cross_entropy_lambda: float = 5.0

    # ---- global projection --------------------------------------------------
    use_projections: bool = True
    projection_alpha: float = 1.0
    projection_g: float = 1.0

    # ---- muon on activations ------------------------------------------------
    use_muon_activations: bool = False
    muon_activations_lr: float = 0.5
    
    # ---- frozen-A weight solve ----------------------------------------------
    frozen_a_weights: bool = False
    frozen_a_g: float = 1.0

    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def update(self, key: str, value) -> None:
        if key not in _PUBLIC_KEYS:
            raise AttributeError(
                f"Unknown config key {key!r}. Valid keys: {sorted(_PUBLIC_KEYS)}"
            )
        with self._lock:
            object.__setattr__(self, key, value)

    def reset(self) -> None:
        with self._lock:
            for f in fields(self):
                if f.default is not MISSING:
                    object.__setattr__(self, f.name, f.default)

    def snapshot(self) -> dict:
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if not f.name.startswith("_")
        }


_PUBLIC_KEYS: frozenset = frozenset(
    f.name for f in fields(Config) if not f.name.startswith("_")
)

config = Config()
