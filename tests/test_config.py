"""Each test proves one Config option does what it claims."""
import pytest
import torch
import torch.nn.functional as F
from dataclasses import fields, MISSING

from ptorch.config import Config, config
from ptorch.core.ops import process_activation_target
from ptorch.nn.modules import (
    Linear, ReLU, LeakyReLU, Softmax, CrossEntropy, MaxPool2d,
)
from ptorch.nn.modules_experimental import (
    Rotary, RMSNorm, apply_rotary_emb,
)


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


DET, PROJ  = _rnd(8, 16, seed=4), _rnd(8, 16, seed=5) * 0.5


# ── Config class behavior ──────────────────────────────────────────────────

def test_defaults_match_dataclass():
    for f in _public_fields():
        assert getattr(config, f.name) == f.default


def test_update_rejects_unknown_key():
    with pytest.raises(AttributeError):
        config.update("no_such_key", 1)


def test_reset_restores_defaults():
    _set(projection_alpha=99.0, use_muon_activations=True)
    config.reset()
    for f in _public_fields():
        assert getattr(config, f.name) == f.default

# ── use_projections ────────────────────────────────────────────────────────

def test_use_projections_false_gives_standard_linear():
    _set(use_projections=False)
    layer = Linear(8, 6, bias=False)
    x = _rnd(4, 8, seed=10)
    assert torch.allclose(layer(x), F.linear(x, layer.weight) / layer.omega, atol=1e-5)


def test_use_projections_false_linear_with_bias_matches_F_linear():
    _set(use_projections=False)
    layer = Linear(8, 6, bias=True)
    x = _rnd(4, 8, seed=11)
    expected = F.linear(x, layer.weight, layer.bias.squeeze(0)) / layer.omega
    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_use_projections_false_relu_matches_F_relu():
    _set(use_projections=False)
    x = _rnd(4, 8, seed=12)
    assert torch.equal(ReLU()(x), F.relu(x))


def test_use_projections_false_leaky_relu_matches_F_leaky_relu():
    _set(use_projections=False)
    x = _rnd(4, 8, seed=13)
    assert torch.equal(LeakyReLU(0.1)(x), F.leaky_relu(x, 0.1))


def test_use_projections_false_softmax_matches_F_softmax():
    _set(use_projections=False)
    x = _rnd(4, 8, seed=14)
    assert torch.allclose(Softmax()(x), F.softmax(x, dim=-1), atol=1e-6)


def test_use_projections_false_cross_entropy_matches_F_cross_entropy():
    _set(use_projections=False)
    logits = _rnd(4, 5, seed=16)
    target = torch.tensor([0, 2, 4, 1])
    assert torch.allclose(CrossEntropy()(logits, target),
                          F.cross_entropy(logits, target), atol=1e-6)


def test_use_projections_false_maxpool2d_matches_F_max_pool2d():
    _set(use_projections=False)
    x = _rnd(2, 3, 8, 8, seed=17)
    assert torch.equal(MaxPool2d(2)(x), F.max_pool2d(x, 2))


def test_use_projections_false_rotary_matches_apply_rotary_emb():
    _set(use_projections=False)
    rot = Rotary(8)
    x = _rnd(2, 2, 4, 8, seed=18)  # (batch, heads, seqlen, head_dim)
    out = rot(x)
    # Reproduce the cos/sin Rotary uses internally
    rot._update_cache(x.size(-2), x.device, x.dtype)
    cos = rot._cos_cached.to(dtype=x.dtype)
    sin = rot._sin_cached.to(dtype=x.dtype)
    assert torch.equal(out, apply_rotary_emb(x, cos, sin))


def test_use_projections_false_rmsnorm_matches_F_rms_norm():
    _set(use_projections=False)
    norm = RMSNorm()
    x = _rnd(4, 8, seed=19)
    assert torch.allclose(norm(x), F.rms_norm(x, (x.size(-1),), eps=norm.eps), atol=1e-6)


def test_use_projections_false_backward_populates_real_gradients():
    """End-to-end smoke test: a small MLP under use_projections=False must
    produce real `weight.grad` via vanilla autograd. Catches future ungated
    layers as soon as anyone wires them into a model."""
    _set(use_projections=False)
    l1 = Linear(8, 16, bias=True)
    l2 = Linear(16, 5, bias=True)
    act = ReLU()
    crit = CrossEntropy()

    x = _rnd(4, 8, seed=20)
    target = torch.tensor([0, 2, 4, 1])
    logits = l2(act(l1(x)))
    loss = crit(logits, target)
    loss.backward()

    assert l1.weight.grad is not None and l1.weight.grad.abs().sum() > 0
    assert l2.weight.grad is not None and l2.weight.grad.abs().sum() > 0


# ── muon on activations ────────────────────────────────────────────────────

def test_muon_activations_off_returns_proj_unchanged():
    _set(use_muon_activations=False)
    out = process_activation_target(DET, PROJ)
    assert torch.allclose(out, PROJ)


def test_muon_activations_lr_zero_returns_det_unchanged():
    _set(use_muon_activations=True, muon_activations_lr=0.0)
    out = process_activation_target(DET, PROJ)
    assert torch.allclose(out, DET, atol=1e-5)




