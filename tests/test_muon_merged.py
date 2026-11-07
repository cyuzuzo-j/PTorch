"""ProjectionMuon runs all matrices of a param group as ONE compiled call
(_muon_update). It must track eager torch.optim.Muon on the same .grad
(up to bf16 Newton-Schulz rounding), compile exactly one
graph per param group, never recompile on lr changes, and round-trip its
state_dict."""
import copy

import pytest
import torch
from torch._dynamo.utils import counters

import ptorch.optim_static as poptim

SHAPES = [(64, 192), (10, 64), (96, 48)]  # wide, wide, tall (transposed NS)
STEPS = 15


@pytest.fixture(autouse=True)
def _reset_compile():
    torch._dynamo.reset()
    counters.clear()


def _params(seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.nn.Parameter(torch.randn(s, generator=g) / s[1] ** 0.5) for s in SHAPES]


def _grads(seed=1):
    g = torch.Generator().manual_seed(seed)
    return [[torch.randn(s, generator=g) for s in SHAPES] for _ in range(STEPS)]


def _groups(ps):  # two groups with different hyperparameters
    return [{"params": ps[:2]}, {"params": ps[2:], "lr": 2e-3, "weight_decay": 0.0}]


def _run(opt, ps, grads, sched=None):
    for gs in grads:
        for p, g in zip(ps, gs):
            p.grad = g.clone()
        opt.step()
        if sched is not None:
            sched.step()


def _rel_err(a, b):
    return max(((x - y).norm() / y.norm()).item() for x, y in zip(a, b))


@pytest.mark.parametrize("schedule", [False, True])
def test_matches_torch_muon(schedule):
    ref, mine = _params(), _params()
    grads = _grads()
    kw = dict(lr=1e-3, weight_decay=0.1, momentum=0.95)
    o_ref = torch.optim.Muon(_groups(ref), **kw)
    o_mine = poptim.ProjectionMuon(_groups(mine), **kw)
    s_ref = s_mine = None
    if schedule:
        s_ref = torch.optim.lr_scheduler.CosineAnnealingLR(o_ref, T_max=STEPS)
        s_mine = torch.optim.lr_scheduler.CosineAnnealingLR(o_mine, T_max=STEPS)
    _run(o_ref, ref, grads, s_ref)
    _run(o_mine, mine, grads, s_mine)
    # bf16 Newton-Schulz: compiled and eager round differently, ~lr-sized drift
    assert _rel_err(mine, ref) < 2e-4
    # one graph per param group (the second group's guard miss is the only
    # "recompile"), none from the scheduled lr
    assert counters["stats"]["unique_graphs"] == 2
    assert counters["frames"]["total"] == 2


def test_state_dict_round_trip():
    ps = _params()
    grads = _grads()
    o = poptim.ProjectionMuon(_groups(ps), lr=1e-3, weight_decay=0.1)
    _run(o, ps, grads[:5])
    sd = copy.deepcopy(o.state_dict())
    snapshot = [p.detach().clone() for p in ps]
    _run(o, ps, grads[5:])
    after = [p.detach().clone() for p in ps]

    ps2 = [torch.nn.Parameter(t.clone()) for t in snapshot]
    o2 = poptim.ProjectionMuon(_groups(ps2), lr=1e-3, weight_decay=0.1)
    o2.load_state_dict(sd)
    _run(o2, ps2, grads[5:])
    for a, b in zip(ps2, after):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
