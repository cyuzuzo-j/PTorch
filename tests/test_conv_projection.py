"""Conv2DProjection and the unpadded max-pool path against their patch-matrix references."""
import pytest
import torch
import torch.nn.functional as F

import ptorch.nn.modules as pnn
from ptorch.config import config
from ptorch.core.ops import ConvPatchProjection, max_proj_pt_batch, plain_solve

DT = torch.float64


@pytest.fixture(autouse=True)
def projections():
    config.reset()  # other test modules leave target overrides set
    config.use_projections = True
    assert plain_solve()
    yield
    config.reset()


def solve(conv, x, z, patches):
    """x's target and the weight and bias residuals for target z (contiguous NCHW)."""
    conv.zero_grad()
    x = x.detach().clone().requires_grad_()
    if patches:
        y = conv.linear(ConvPatchProjection.apply(x, conv.kernel_size, conv.stride, conv.padding)).permute(0, 3, 1, 2)
    else:
        y = conv(x)
        assert type(y.grad_fn).__name__ == "Conv2DProjectionBackward"
    y.backward(z)
    bias = None if conv.linear.bias is None else conv.linear.bias.grad.clone()
    return y.detach(), x.grad, conv.linear.weight.grad.clone(), bias


@pytest.mark.parametrize("k,s,p,bias,alpha_auto", [
    (3, 1, 1, True, False), (3, 2, 1, True, False), (1, 2, 0, False, False),
    (5, 1, 2, True, True), (2, 2, 0, True, False), (3, 3, 0, True, False),  # H=7, k=s=3: a pixel row is uncovered
])
def test_conv_matches_patch_solve(k, s, p, bias, alpha_auto):
    config.update("projection_alpha_auto", alpha_auto)
    torch.manual_seed(0)
    conv = pnn.Conv2D(3, 5, k, s, p, bias=bias).to(DT)
    x = torch.randn(4, 3, 7, 7, dtype=DT)
    y = conv(x).detach()
    z = y + 0.5 * torch.randn(y.shape, dtype=DT)
    for fast, ref in zip(solve(conv, x, z, False), solve(conv, x, z, True)):
        if ref is not None:
            torch.testing.assert_close(fast, ref, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("h,k", [(8, 2), (9, 2), (11, 4), (12, 3)])
def test_maxpool_matches_unfold(h, k):
    torch.manual_seed(0)
    x = torch.randn(2, 3, h, h, dtype=DT, requires_grad=True)
    y = pnn.MaxPool2d(k)(x)
    z = y.detach() + torch.randn(y.shape, dtype=DT)
    y.backward(z)

    a = F.unfold(x.detach(), k, stride=k).view(2, 3, k * k, -1).permute(0, 1, 3, 2).reshape(-1, k * k)
    a = max_proj_pt_batch(a, z.reshape(-1, 1)).view(2, 3, -1, k * k).permute(0, 1, 3, 2).reshape(2, 3 * k * k, -1)
    ref = x.detach().clone()
    c = (h // k) * k
    ref[:, :, :c, :c] = F.fold(a, (c, c), k, stride=k)
    torch.testing.assert_close(x.grad, ref)
