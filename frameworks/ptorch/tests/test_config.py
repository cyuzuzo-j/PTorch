"""Each test proves one Config option does what it claims."""
import pytest
import torch
import torch.nn.functional as F
from dataclasses import fields, MISSING

from frameworks.ptorch.config import Config, config
from frameworks.ptorch.core.ops import (
    MatMulProjection, MatMulProjectionFrozenA,
    process_activation_target, process_weight_target,
)
from frameworks.ptorch.nn.modules import Linear


# ── fixtures & helpers ─────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_config():
    config.reset()


def _rnd(*shape, seed):
    g = torch.Generator(); g.manual_seed(seed)
    return torch.randn(*shape, generator=g)


def _set(**kw):
    for k, v in kw.items():
        config.update(k, v)


def _public_fields():
    return [f for f in fields(Config)
            if not f.name.startswith("_") and f.default is not MISSING]


A, B, TGT  = _rnd(8, 6, seed=1), _rnd(6, 5, seed=2), _rnd(8, 5, seed=3)
DET, PROJ  = _rnd(8, 16, seed=4), _rnd(8, 16, seed=5) * 0.5


def _matmul_b_grad(alpha=1.0, g=1.0):
    """B.grad through MatMulProjection (consults config in backward)."""
    a, b = A.detach().requires_grad_(True), B.detach().requires_grad_(True)
    MatMulProjection.apply(a, b, 5, alpha, g, 1.0, None, False, False, None).backward(TGT)
    return b.grad.clone()


def _frozen_b_grad():
    """B.grad through MatMulProjectionFrozenA (consults config in backward)."""
    a, b = A.detach().requires_grad_(True), B.detach().requires_grad_(True)
    MatMulProjectionFrozenA.apply(a, b, 1.0, 1.0, False, None).backward(TGT)
    return b.grad.clone()


# ── Config class behavior ──────────────────────────────────────────────────

def test_defaults_match_dataclass():
    for f in _public_fields():
        assert getattr(config, f.name) == f.default


def test_update_rejects_unknown_key():
    with pytest.raises(AttributeError):
        config.update("no_such_key", 1)


def test_reset_restores_defaults():
    _set(projection_alpha=99.0, muon_weights=True)
    config.reset()
    for f in _public_fields():
        assert getattr(config, f.name) == f.default

# ── use_projections ────────────────────────────────────────────────────────

def test_use_projections_false_gives_standard_linear():
    _set(use_projections=False)
    layer = Linear(8, 6, bias=False)
    x = _rnd(4, 8, seed=10)
    assert torch.allclose(layer(x), F.linear(x, layer.weight) / layer.omega, atol=1e-5)


# ── projection knobs (alpha, g) ────────────────────────────────────────────

@pytest.mark.parametrize("knob", ["alpha", "g"])
def test_projection_knob_changes_b_grad(knob):
    _set(frozen_a_weights=False)
    lo = _matmul_b_grad(**{knob: 0.1})
    hi = _matmul_b_grad(**{knob: 5.0})
    assert not torch.allclose(lo, hi, atol=1e-4)


# ── muon on activations ────────────────────────────────────────────────────

def test_muon_activations_off_returns_proj_unchanged():
    _set(use_muon_activations=False)
    assert torch.allclose(process_activation_target(DET, PROJ), PROJ)


def test_muon_activations_lr_zero_returns_det_unchanged():
    _set(use_muon_activations=True, muon_activations_lr=0.0)
    assert torch.allclose(process_activation_target(DET, PROJ), DET, atol=1e-5)




# ── muon on weights ────────────────────────────────────────────────────────

def test_muon_weights_off_returns_proj_unchanged():
    _set(muon_weights=False)
    det, proj = _rnd(4, 8, seed=6), _rnd(4, 8, seed=7)
    assert torch.allclose(process_weight_target(det, proj), proj)


def test_muon_weights_lr_zero_returns_det_unchanged():
    _set(muon_weights=True, muon_weights_lr=0.0)
    det, proj = _rnd(4, 8, seed=8), _rnd(4, 8, seed=9)
    assert torch.allclose(process_weight_target(det, proj), det, atol=1e-5)


def test_muon_weights_scale_preserves_gradient_norm():
    _set(muon_weights=True, muon_weights_lr=1.0, muon_weights_scale=True)
    det, proj = _rnd(4, 8, seed=10), _rnd(4, 8, seed=11)
    out = process_weight_target(det, proj)
    assert abs((out - det).norm() - (det - proj).norm()) < 1e-3


# ── frozen-A solve ─────────────────────────────────────────────────────────

def test_frozen_a_weights_toggles_b_grad():
    _set(muon_weights=False, frozen_a_weights=False)
    free = _matmul_b_grad()
    _set(frozen_a_weights=True)
    frozen = _matmul_b_grad()
    assert not torch.allclose(free, frozen, atol=1e-4)


def test_frozen_a_g_scales_b_grad():
    _set(muon_weights=False, frozen_a_g=0.1)
    lo = _frozen_b_grad()
    _set(frozen_a_g=10.0)
    hi = _frozen_b_grad()
    assert not torch.allclose(lo, hi, atol=1e-4)


# muon_weights must remain effective in both frozen-A code paths
@pytest.mark.parametrize("b_grad", [_matmul_b_grad, _frozen_b_grad],
                         ids=["MatMulProjection+frozen_a", "MatMulProjectionFrozenA"])
def test_muon_weights_active_in_frozen_paths(b_grad):
    _set(frozen_a_weights=True, muon_weights=False)
    off = b_grad()
    _set(muon_weights=True)
    on = b_grad()
    assert not torch.allclose(off, on, atol=1e-4)
