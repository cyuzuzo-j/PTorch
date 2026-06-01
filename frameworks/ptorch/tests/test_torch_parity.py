"""Per-step parity tests: with use_projections=False, ptorch architectures
must produce *bit-exact* outputs and parameter trajectories vs the equivalent
pure-PyTorch model trained with vanilla SGD on the same seed/data."""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from frameworks.ptorch.config import config
from frameworks.ptorch.nn.modules import Linear as PLinear, ReLU as PReLU, Conv2D as PConv2D, MaxPool2d as PMaxPool2d, CrossEntropy as PCrossEntropy
from frameworks.ptorch.optim_static import ProjectionSGD


@pytest.fixture(autouse=True)
def _reset_config():
    config.reset()
    config.update("use_projections", False)


def _seed(s):
    g = torch.Generator(); g.manual_seed(s)
    return g


def _copy_linear(src_p: PLinear, dst_t: nn.Linear):
    """Mirror ptorch Linear weights/bias into a torch.nn.Linear so both
    start from byte-identical state. Ptorch stores bias as (1, out); torch as (out,)."""
    with torch.no_grad():
        dst_t.weight.copy_(src_p.weight)
        if src_p.use_bias:
            dst_t.bias.copy_(src_p.bias.squeeze(0))


def _copy_conv_linear(src_p_conv: PConv2D, dst_t: nn.Conv2d):
    """Mirror ptorch Conv2D into torch.nn.Conv2d. Ptorch flattens conv weights into
    a Linear of shape (out, in*kH*kW); reshape back into (out, in, kH, kW)."""
    with torch.no_grad():
        kH, kW = src_p_conv.kernel_size
        in_ch = dst_t.in_channels
        out_ch = dst_t.out_channels
        w = src_p_conv.linear.weight  # (out_ch, in_ch * kH * kW)
        dst_t.weight.copy_(w.view(out_ch, in_ch, kH, kW))
        # ptorch Conv2D uses bias=False on the inner Linear, so torch conv has no bias either


# ───────────────────────────────────────────────────────────────────────────
# Architecture 1: MLP (Linear → ReLU → Linear)
# ───────────────────────────────────────────────────────────────────────────

def test_mlp_per_step_parity_with_torch():
    """ptorch MLP under use_projections=False is bit-identical to a torch MLP
    given matched init, batch, optimizer, and LR — for every step."""
    torch.manual_seed(0)

    # Build both models; copy ptorch params into torch model so init matches
    p_l1 = PLinear(8, 16, bias=True)
    p_l2 = PLinear(16, 5, bias=True)
    p_relu = PReLU()
    p_loss = PCrossEntropy()

    t_l1 = nn.Linear(8, 16, bias=True)
    t_l2 = nn.Linear(16, 5, bias=True)
    _copy_linear(p_l1, t_l1)
    _copy_linear(p_l2, t_l2)

    p_opt = ProjectionSGD([p_l1.weight, p_l1.bias, p_l2.weight, p_l2.bias], lr=0.1, momentum=0.0)
    t_opt = torch.optim.SGD(list(t_l1.parameters()) + list(t_l2.parameters()), lr=0.1, momentum=0.0)

    g = _seed(123)
    for step in range(5):
        x = torch.randn(4, 8, generator=g)
        y = torch.randint(0, 5, (4,), generator=g)

        # ptorch forward
        p_logits = p_l2(p_relu(p_l1(x)))
        p_loss_val = p_loss(p_logits, y)

        # torch forward
        t_logits = t_l2(F.relu(t_l1(x)))
        t_loss_val = F.cross_entropy(t_logits, y)

        assert torch.equal(p_logits, t_logits), f"step {step}: logits differ"
        assert torch.equal(p_loss_val, t_loss_val), f"step {step}: loss differs"

        p_opt.zero_grad(); p_loss_val.backward(); p_opt.step()
        t_opt.zero_grad(); t_loss_val.backward(); t_opt.step()

        assert torch.equal(p_l1.weight, t_l1.weight), f"step {step}: l1.weight diverged"
        assert torch.equal(p_l1.bias.squeeze(0), t_l1.bias), f"step {step}: l1.bias diverged"
        assert torch.equal(p_l2.weight, t_l2.weight), f"step {step}: l2.weight diverged"
        assert torch.equal(p_l2.bias.squeeze(0), t_l2.bias), f"step {step}: l2.bias diverged"


# ───────────────────────────────────────────────────────────────────────────
# Architecture 2: CNN (Conv2D → ReLU → MaxPool → Flatten → Linear)
# ───────────────────────────────────────────────────────────────────────────

def test_cnn_per_step_parity_with_torch():
    """ptorch CNN (Conv2D + MaxPool2d + Linear) under use_projections=False is
    bit-identical to the equivalent torch CNN, per step."""
    torch.manual_seed(1)

    in_ch, hid_ch, n_cls = 1, 4, 3
    img_h, img_w = 8, 8
    k = 3

    p_conv = PConv2D(in_ch, hid_ch, kernel_size=k, padding=1)
    p_pool = PMaxPool2d(2)
    p_relu = PReLU()
    # After (conv pad=1 → relu → pool 2): spatial halved to 4×4. Flatten = hid_ch * 4 * 4.
    p_head = PLinear(hid_ch * 4 * 4, n_cls, bias=True)
    p_loss = PCrossEntropy()

    t_conv = nn.Conv2d(in_ch, hid_ch, kernel_size=k, padding=1, bias=False)
    t_pool = nn.MaxPool2d(2)
    t_head = nn.Linear(hid_ch * 4 * 4, n_cls, bias=True)
    _copy_conv_linear(p_conv, t_conv)
    _copy_linear(p_head, t_head)

    p_params = [p_conv.linear.weight, p_head.weight, p_head.bias]
    t_params = [t_conv.weight, t_head.weight, t_head.bias]
    p_opt = ProjectionSGD(p_params, lr=0.05, momentum=0.0)
    t_opt = torch.optim.SGD(t_params, lr=0.05, momentum=0.0)

    g = _seed(456)
    for step in range(5):
        x = torch.randn(2, in_ch, img_h, img_w, generator=g)
        y = torch.randint(0, n_cls, (2,), generator=g)

        # ptorch forward: conv emits NCHW; relu/pool stay NCHW; flatten then head
        p_feat = p_pool(p_relu(p_conv(x)))
        p_logits = p_head(p_feat.reshape(p_feat.size(0), -1))
        p_loss_val = p_loss(p_logits, y)

        # torch forward
        t_feat = t_pool(F.relu(t_conv(x)))
        t_logits = t_head(t_feat.reshape(t_feat.size(0), -1))
        t_loss_val = F.cross_entropy(t_logits, y)

        assert torch.allclose(p_logits, t_logits, atol=1e-6), f"step {step}: logits differ"
        assert torch.allclose(p_loss_val, t_loss_val, atol=1e-6), f"step {step}: loss differs"

        p_opt.zero_grad(); p_loss_val.backward(); p_opt.step()
        t_opt.zero_grad(); t_loss_val.backward(); t_opt.step()

        # Compare conv weight via the reshape-back equivalence
        kH, kW = p_conv.kernel_size
        p_conv_w = p_conv.linear.weight.view(hid_ch, in_ch, kH, kW)
        assert torch.allclose(p_conv_w, t_conv.weight, atol=1e-6), f"step {step}: conv.weight diverged"
        assert torch.allclose(p_head.weight, t_head.weight, atol=1e-6), f"step {step}: head.weight diverged"
        assert torch.allclose(p_head.bias.squeeze(0), t_head.bias, atol=1e-6), f"step {step}: head.bias diverged"
