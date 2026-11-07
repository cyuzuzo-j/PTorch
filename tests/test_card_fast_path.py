"""MatMulProjection's one-call card solve (_card_proj, bias folded analytically)
must give the same residuals as the general path (bias as a padded ones column)."""
import pytest
import torch
import torch.nn as tnn
import torch.nn.functional as F

import ptorch.core.ops as ops
import ptorch.nn.modules as pnn
from ptorch.config import config


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("alpha_auto", [False, True])
def test_card_fast_path_matches_general(bias, alpha_auto, monkeypatch):
    config.update("projection_alpha_auto", alpha_auto)
    try:
        torch.manual_seed(0)
        m = tnn.Sequential(pnn.Linear(96, 128, bias=bias), pnn.ReLU(), pnn.Linear(128, 10, bias=bias))
        x = torch.randn(3, 64, 96)  # batched rows exercise the (..., K) flatten
        y = F.one_hot(torch.randint(0, 10, (3, 64)), 10).float()
        grads = []
        for fast in (True, False):
            if not fast:
                monkeypatch.setattr(ops, "_plain_card", lambda *a: False)
            m.zero_grad()
            pnn.CrossEntropy()(m(x), y).sum().backward()
            grads.append([p.grad.clone() for p in m.parameters()])
        for g_fast, g_general in zip(*grads):
            torch.testing.assert_close(g_fast, g_general, rtol=1e-5, atol=1e-5)
    finally:
        config.update("projection_alpha_auto", False)
