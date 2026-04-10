import threading
import contextlib
from typing import Any, Optional

# default configuration
defaults = {
    "bilinear_projection_num_newton_steps": 10,
    "cross_entropy_method": "fixed_point",
    "cross_entropy_num_steps": 10,
    "cross_entropy_lambda": 5.0,
    "use_projections": True,
    "projection_alpha": 1.0,
    "projection_g": 1.0,
    "projection_norm": "inf",
    "projection_p": 2.0,
    "muon_activations": True,
    "muon_activations_lr": 1.0,
    "muon_activations_scale": False,
}


class Config:
    """Global configuration singleton for the PJAX framework.

    Thread-safe access to algorithmic parameters like projection
    step counts and cross-entropy solver settings.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self):
        self._config = defaults.copy()

    def update(self, key: str, value: Any) -> None:
        """Update a configuration value."""
        with self._lock:
            self._config[key] = value

    def reset(self):
        self._init()

    def __getattr__(self, name: str) -> Any:
        """Access a configuration value by attribute."""
        return self._config[name]

    def __getitem__(self, name: str) -> Any:
        """Access a configuration value by key."""
        return self._config[name]

    def snapshot(self) -> dict:
        """Return a copy of the current configuration as a plain dict."""
        return dict(self._config)

    @contextlib.contextmanager
    def projections(self, enabled: bool, norm: Optional[str] = None):
        """Context manager to toggle projection-based gradients."""
        with self._lock:
            prev_enabled = self._config.get("use_projections", True)
            prev_norm = self._config.get("projection_norm", "l2")
            self._config["use_projections"] = enabled
            if norm is not None:
                self._config["projection_norm"] = norm.lower()
        try:
            yield
        finally:
            with self._lock:
                self._config["use_projections"] = prev_enabled
                if norm is not None:
                    self._config["projection_norm"] = prev_norm

config = Config()


def __getattr__(name: str) -> Any:
    """Access a configuration value by attribute."""
    return getattr(config, name)


def __getitem__(name: str) -> Any:
    """Access a configuration value by key."""
    return config[name]


def update(key: str, value: Any) -> None:
    """Update a configuration value."""
    config.update(key, value)
