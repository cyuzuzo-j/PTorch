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

    # ---- bilinear projection ------------------------------------------------
    bilinear_projection_num_newton_steps: int = 10

    # ---- cross-entropy constraint -------------------------------------------
    cross_entropy_method: str = "fixed_point"
    cross_entropy_num_steps: int = 10
    cross_entropy_lambda: float = 5.0

    # ---- global projection --------------------------------------------------
    use_projections: bool = True
    projection_norm: str = "l2"
    projection_alpha: float = 1.0
    projection_g: float = 1.0
    projection_p: Optional[float] = None

    # ---- muon on activations ------------------------------------------------
    muon_activations: bool = True
    muon_activations_lr: float = 0.5
    muon_activations_scale: bool = False
    muon_activations_norm_preserve: bool = False

    # ---- muon on weights ----------------------------------------------------
    muon_weights: bool = False
    muon_weights_lr: float = 0.02
    muon_weights_scale: bool = False

    # ---- frozen-A weight solve ----------------------------------------------
    frozen_a_weights: bool = True
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

    @contextlib.contextmanager
    def projections(self, enabled: bool, norm: Optional[str] = None):
        prev_enabled = self.use_projections
        prev_norm = self.projection_norm
        self.update("use_projections", enabled)
        if norm is not None:
            self.update("projection_norm", norm.lower())
        try:
            yield
        finally:
            self.update("use_projections", prev_enabled)
            if norm is not None:
                self.update("projection_norm", prev_norm)


_PUBLIC_KEYS: frozenset = frozenset(
    f.name for f in fields(Config) if not f.name.startswith("_")
)

config = Config()
