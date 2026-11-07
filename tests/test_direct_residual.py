"""Every parameter's .grad holds the residual theta - theta_bar, never the target
theta_bar: each case below runs one backward and returns its parameters with
independently computed fp64 targets. Activation operands keep receiving targets,
because autograd passes targets between nodes."""
import copy
import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn as tnn
import torch.nn.functional as F

import ptorch
import ptorch.core.ops as ops
import ptorch.nn.modules as pnn
import ptorch.nn.modules_experimental as pxm
from ptorch.config import config
from ptorch.optim_static import ProjectionAdam, ProjectionMuon, ProjectionSGD

D = torch.float64
_spec = importlib.util.spec_from_file_location(
    "unfused_linear", Path(__file__).parents[1] / "experiments/neurips/mlp/unfused_linear.py")
unfused = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(unfused)


@pytest.fixture(autouse=True)
def _reset_config():
    config.reset()
    yield
    config.reset()


def _data(*shape):
    return torch.randn(*shape, dtype=D)


# ── cases: run a backward, return [(parameter, its value before, its target)] ──

def _linear_l2():
    lin = pnn.Linear(12, 9, num_iters=20).double()
    x, z = _data(16, 12), _data(16, 9)
    w0, b0 = lin.weight.detach().clone(), lin.bias.detach().clone()
    lin(x).backward(z)
    _, Bt, _, _, bt = ops._matmul_proj_core(x, w0.T, z, num_steps=20, bias=b0)
    return [(lin.weight, w0, Bt.T), (lin.bias, b0, bt)]


def _linear_linf():
    lin = pnn.Linear(6, 4, norm="linf", num_iters=10).double()
    x, z = _data(8, 6), _data(8, 4)
    w0, b0 = lin.weight.detach().clone(), lin.bias.detach().clone()
    lin(x).backward(z)
    B = torch.cat([w0.T, b0])
    _, Bt, _, _ = ops.matmul_proj_linf(F.pad(x, (0, 1), value=1.0), B, z, num_steps=10)
    return [(lin.weight, w0, Bt[:-1].T), (lin.bias, b0, Bt[-1:])]


def _conv2d():
    conv = pnn.Conv2D(2, 3, kernel_size=3, padding=1, num_iters=7).double()
    x = _data(2, 2, 5, 5)
    lin = conv.linear
    w0, b0 = lin.weight.detach().clone(), lin.bias.detach().clone()
    out = conv(x)
    z = out.detach() + 0.5 * torch.randn_like(out)
    out.backward(z)
    patches = F.unfold(x, 3, padding=1).transpose(1, 2).reshape(-1, 18)
    _, Bt, _, _, bt = ops._matmul_proj_core(patches, w0.T, z.permute(0, 2, 3, 1).reshape(-1, 3),
                                            num_steps=7, bias=b0)
    return [(lin.weight, w0, Bt.T), (lin.bias, b0, bt)]


def _scale():
    m = pxm.Scale(5, init=0.7).double()
    x = _data(4, 3, 5)
    x[..., 0] = 0.0  # empty column: the eps clamp of the least-squares fit
    z = _data(4, 3, 5)
    a0 = m.weight.detach().clone()
    m(x).backward(z)
    fit = (z * x).sum((0, 1)) / (x * x).sum((0, 1)).clamp(min=1e-8)
    return [(m.weight, a0, (a0 + fit) / 2)]


def _embedding(transit=False):
    config.update("transit_vjp_embed", transit)
    m = pxm.Embedding(6, 4, init_std=0.5).double()
    ids = torch.tensor([[0, 2, 2], [5, 2, 0]])  # rows 1, 3, 4 stay untouched
    z = _data(2, 3, 4)
    w0 = m.weight.detach().clone()
    m(ids).backward(z)
    target = w0.clone()
    for v in ids.unique():
        d = z[ids == v] - w0[v]
        target[v] += d.sum(0) * (1.0 if transit else len(d) ** (config.embed_count_beta - 1.0))
    return [(m.weight, w0, target)]


def _log_std(shape=(3,), init=-0.4, steps=3):
    log_std = tnn.Parameter(torch.full(shape, init, dtype=D))
    mu = _data(5, 3).requires_grad_()
    act, w = _data(5, 3), _data(5).abs()
    ops.GaussianPolicyGradientProjection.apply(mu, log_std, act, w, steps, 0.2).backward()
    ls, mu_t = log_std.detach().clone(), mu.detach()
    cls = ops.GaussianPolicyGradientProjection
    for _ in range(steps):
        std = torch.exp(ls)
        ls_new = ls + 0.2 * (w[:, None] * (((act - mu_t) / std) ** 2 - 1.0)).mean(0)
        mu_t = mu_t + 0.2 * w[:, None] * (act - mu_t) / std ** 2
        ls = ls_new.clamp(cls.LOG_STD_MIN, cls.LOG_STD_MAX)
    torch.testing.assert_close(mu.grad, mu_t)  # mu is an activation: its target
    return [(log_std, torch.full(shape, init, dtype=D), ls)]


CASES = {
    "linear_l2": _linear_l2,
    "linear_linf": _linear_linf,
    "conv2d": _conv2d,
    "scale": _scale,
    "embedding": _embedding,
    "embedding_transit": lambda: _embedding(transit=True),
    "log_std": _log_std,
    "log_std_row": lambda: _log_std(shape=(1, 3)),
    "log_std_clamped": lambda: _log_std(init=1.99),
}


@pytest.mark.parametrize("case", CASES)
def test_grad_is_the_residual(case):
    torch.manual_seed(0)
    for p, before, target in CASES[case]():
        torch.testing.assert_close(p.grad, before - target, rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("case", CASES)
def test_sgd_lr1_lands_on_the_target(case):
    torch.manual_seed(0)
    entries = CASES[case]()
    ProjectionSGD([p for p, _, _ in entries], lr=1.0).step()
    for p, _, target in entries:
        torch.testing.assert_close(p.detach(), target, rtol=1e-9, atol=1e-11)


# ── the numerical point: no fp32 cancellation ─────────────────────────────────

@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("alpha,g", [(1.0, 1.0), (0.5, 2.0)])
@pytest.mark.parametrize("deflate", [False, True])
def test_core_residual_equals_fp64_subtraction(bias, alpha, g, deflate):
    torch.manual_seed(0)
    A, B, Z = _data(16, 12), 0.4 * _data(12, 9), 2 * _data(16, 9)
    b = 0.3 * _data(1, 9) if bias else None
    kw = dict(alpha=alpha, g=g, num_steps=20, b_self_deflate=deflate, bias=b)
    _, Bt, _, _, bt = ops._matmul_proj_core(A, B, Z, **kw)
    _, Br, _, _, br = ops._matmul_proj_core(A, B, Z, direct_residual=True, **kw)
    torch.testing.assert_close(Br, B - Bt, rtol=1e-10, atol=1e-12)
    if bias:
        torch.testing.assert_close(br, b - bt, rtol=1e-10, atol=1e-12)


def _rel(a, ref):
    return float((a.double() - ref).norm() / ref.norm())


@pytest.mark.parametrize("norm", ["l2", "linf"])
def test_fp32_keeps_a_residual_at_weight_resolution(norm):
    """Small activations make the residual tiny next to W (1e-7 of ||W|| for l2,
    1e-4 for l_inf) with the output misfit still as large as the output, the
    first-layer regime. Subtracting an fp32 target loses much of it; the
    reference is fp64 on the same fp32 operands."""
    torch.manual_seed(0)
    s = 1e-3
    x = s * torch.randn(64, 8)
    lin = pnn.Linear(8, 4, bias=False, norm=norm, num_iters=20)
    w = lin.weight.detach().clone()
    out = lin(x)
    z = out.detach() + s * torch.randn(64, 4)
    out.backward(z)
    solve = ops.matmul_proj_linf if norm == "linf" else ops._matmul_proj_core
    ref = w.double() - solve(x.double(), w.double().T, z.double(), num_steps=20)[1].T
    assert float(ref.norm() / w.double().norm()) < 1e-3
    direct = _rel(lin.weight.grad, ref)
    assert direct < 1e-5
    assert _rel(w - solve(x, w.T, z, num_steps=20)[1].T, ref) > 100 * direct


def test_fp32_scale_residual():
    torch.manual_seed(0)
    m = pxm.Scale(8, init=1.0)  # a = 1: the forward product is exact
    x = torch.randn(64, 8)
    out = m(x)
    z = out.detach() + 3e-7 * torch.randn(64, 8)
    out.backward(z)
    x64, z64 = x.double(), z.double()
    ref = (x64 * (x64 - z64)).sum(0) / (x64 * x64).sum(0) / 2
    assert _rel(m.weight.grad, ref) < 1e-4
    assert _rel(1.0 - (1.0 + (z * x).sum(0) / (x * x).sum(0)) / 2, ref) > 0.1


# ── tied embedding: residuals are additive ────────────────────────────────────

def _tied_model(tied):
    emb, head = pxm.Embedding(7, 5, init_std=0.5).double(), pnn.Linear(5, 7, bias=False).double()
    head.weight = emb.weight
    emb.tied = tied
    return emb, head


def _untied_residuals(ids, z):
    """The same network, with the head on its own copy of the shared weight."""
    emb, head = _tied_model(False)
    head.weight = tnn.Parameter(emb.weight.detach().clone())
    head(emb(ids)).backward(z)
    return head.weight.grad, emb.weight.grad, emb.weight.detach().clone()


def test_tied_weight_sums_or_splits_the_two_residuals():
    ids = torch.tensor([[0, 1, 1], [4, 6, 0]])
    z = _data(2, 3, 7)
    state = torch.get_rng_state()
    d_head, d_emb, w0 = _untied_residuals(ids, z)

    torch.set_rng_state(state)
    emb, head = _tied_model(False)  # plain sharing: autograd adds them
    head(emb(ids)).backward(z)
    torch.testing.assert_close(emb.weight.grad, d_head + d_emb)

    torch.set_rng_state(state)
    emb, head = _tied_model(True)  # split for the training loop's combine rule
    head(emb(ids)).backward(z)
    torch.testing.assert_close(emb.weight.grad, d_head)
    torch.testing.assert_close(emb.tied_sink[0], d_emb)

    emb.weight.grad = 0.5 * (emb.weight.grad + emb.tied_sink[0])
    ProjectionSGD([emb.weight], lr=1.0).step()
    torch.testing.assert_close(emb.weight.detach(), 0.5 * ((w0 - d_head) + (w0 - d_emb)))


# ── the unfused ablation row ──────────────────────────────────────────────────

@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-10), (torch.float32, 1e-3)])
@pytest.mark.parametrize("scale", [1.0, 1e-3])  # 1e-3: residuals near 1e-7 of ||W||
def test_unfused_matches_fused_linear(dtype, tol, scale):
    torch.manual_seed(0)
    fused = pnn.Linear(12, 9).to(dtype)
    mine = unfused.UnfusedLinear(12, 9).to(dtype)
    mine.load_state_dict(fused.state_dict())
    x, dz = scale * torch.randn(16, 12, dtype=dtype), scale * torch.randn(16, 9, dtype=dtype)
    xs = [x.clone().requires_grad_() for _ in range(2)]
    for m, xi in zip((fused, mine), xs):
        out = m(xi)
        out.backward(out.detach() + dz)
    assert _rel(mine.weight.grad, fused.weight.grad.double()) < tol
    assert _rel(mine.bias.grad, fused.bias.grad.double()) < tol
    assert _rel(xs[1].grad, xs[0].grad.double()) < tol


# ── activation operands keep their targets ────────────────────────────────────

def test_pairwise_operands_get_targets():
    torch.manual_seed(0)
    A, B, z = _data(8, 6).requires_grad_(), _data(6, 5).requires_grad_(), _data(8, 5)
    ops.MatMulProjection.apply(A, B, 5, 1.0, 1.0, 1.0, True).backward(z)
    At, Bt = ops._matmul_proj_core(A.detach(), B.detach(), z, num_steps=5)[:2]
    torch.testing.assert_close(A.grad, At)
    torch.testing.assert_close(B.grad, Bt)


@pytest.mark.parametrize("norm", ["l2", "linf"])
def test_linear_input_gets_its_target(norm):
    torch.manual_seed(0)
    lin = pnn.Linear(6, 4, bias=False, norm=norm, num_iters=5).double()
    x, z = _data(8, 6).requires_grad_(), _data(8, 4)
    lin(x).backward(z)
    W = lin.weight.detach().T
    solve = ops.matmul_proj_linf if norm == "linf" else ops._matmul_proj_core
    torch.testing.assert_close(x.grad, solve(x.detach(), W, z, num_steps=5)[0])


def test_scale_input_gets_its_target():
    torch.manual_seed(0)
    m = pxm.Scale(5, init=0.7).double()
    x, z = _data(4, 5).requires_grad_(), _data(4, 5)
    m(x).backward(z)
    a = 0.7
    torch.testing.assert_close(x.grad, x.detach() + a * (z - a * x.detach()) / (1 + a * a))


def test_attention_block_trains_on_residuals():
    """A pairwise solve sits below the output projection and above c_q/c_k/c_v:
    targets must cross it, and the weights must come out as small residuals."""
    torch.manual_seed(0)
    attn = pxm.CausalSelfAttention(16, 2, 2, 10000.0, 1.0)
    x = torch.randn(2, 6, 16, requires_grad=True)
    out = attn(x)
    out.backward(out.detach() + 1e-3 * torch.randn_like(out))
    for p in attn.parameters():
        assert p.grad is not None and p.grad.norm() < 0.1 * p.detach().norm()
    assert (x.grad - x.detach()).norm() < 0.5 * x.detach().norm()  # a target: near x


@pytest.mark.parametrize("op", [lambda a, b: a + b, lambda a, b: b - a, lambda a, b: a * b,
                                lambda a, b: a.square() + b, lambda a, b: a.sum() + b])
def test_overridden_tensor_ops(op):
    """A leaf tensor is an activation and gets its target; a Parameter holding
    the same value gets the residual."""
    torch.manual_seed(0)
    v, b, z = _data(4), _data(4), _data(4)
    leaf, param = v.clone().requires_grad_(), tnn.Parameter(v.clone())
    op(leaf, b).backward(z)
    op(param, b).backward(z)
    torch.testing.assert_close(param.grad, v - leaf.grad)


# ── .grad behaves like a gradient buffer ──────────────────────────────────────

def test_residuals_accumulate_in_a_stable_buffer():
    torch.manual_seed(0)
    lin = pnn.Linear(6, 4)
    x, z = torch.randn(8, 6), torch.randn(8, 4)
    lin(x).backward(z)
    single, buf = lin.weight.grad.clone(), lin.weight.grad
    lin(x).backward(z)
    torch.testing.assert_close(lin.weight.grad, 2 * single)
    opt = ProjectionSGD(lin.parameters(), lr=0.0)
    opt.zero_grad(set_to_none=False)
    lin(x).backward(z)
    assert lin.weight.grad.data_ptr() == buf.data_ptr()
    torch.testing.assert_close(lin.weight.grad, single)


def test_deepcopy_keeps_the_convention():
    torch.manual_seed(0)
    lin = pnn.Linear(3, 2)
    other = copy.deepcopy(lin)
    x, z = torch.randn(5, 3), torch.randn(5, 2)
    lin(x).backward(z)
    other(x).backward(z)
    torch.testing.assert_close(lin.weight.grad, other.weight.grad)


def test_general_path_matches_card_path(monkeypatch):
    grads = []
    for fast in (True, False):
        if not fast:
            monkeypatch.setattr(ops, "_plain_card", lambda *a: False)
        torch.manual_seed(0)
        lin = pnn.Linear(12, 9)
        lin(torch.randn(16, 12)).backward(torch.randn(16, 9))
        grads.append((lin.weight.grad.clone(), lin.bias.grad.clone()))
    for a, b in zip(*grads):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)


# ── gradient mode ─────────────────────────────────────────────────────────────

def test_gradient_mode_holds_true_gradients():
    config.update("use_projections", False)
    torch.manual_seed(0)
    lin = pnn.Linear(6, 4)
    scale = pxm.Scale(4, init=0.5)
    x, g = torch.randn(8, 6), torch.randn(8, 4)
    w0, a0 = lin.weight.detach().clone(), scale.weight.detach().clone()
    h = lin(x)
    scale(h).backward(g)
    torch.testing.assert_close(scale.weight.grad, (g * h.detach()).sum(0))
    torch.testing.assert_close(lin.weight.grad, (g * a0).T @ x)
    torch.testing.assert_close(lin.bias.grad, (g * a0).sum(0, keepdim=True))
    ProjectionSGD([lin.weight, scale.weight], lr=0.1).step()
    torch.testing.assert_close(lin.weight.detach(), w0 - 0.1 * lin.weight.grad)
    torch.testing.assert_close(scale.weight.detach(), a0 - 0.1 * scale.weight.grad)


def test_conversion_bridge_feeds_a_plain_torch_layer():
    """Conversion turns the target into a true gradient for the layer below,
    which the Projection* optimizers apply like any other .grad."""
    torch.manual_seed(0)
    lin = tnn.Linear(6, 4)
    x, z = torch.randn(8, 6), torch.randn(8, 4)
    h = lin(x)
    pxm.Conversion()(h).backward(z)
    g = h.detach() - z
    torch.testing.assert_close(lin.weight.grad, g.T @ x)
    w0 = lin.weight.detach().clone()
    ProjectionSGD([lin.weight], lr=0.1).step()
    torch.testing.assert_close(lin.weight.detach(), w0 - 0.1 * lin.weight.grad)


def test_compiled_optimizer_step_matches_eager():
    states = []
    for compiled in (False, True):
        torch._dynamo.reset()
        torch.manual_seed(0)
        m = tnn.Sequential(pnn.Linear(20, 32), pnn.ReLU(), pnn.Linear(32, 5))
        o = ProjectionMuon(m.parameters(), lr=1e-2)
        if compiled:
            o.step = torch.compile(o.step)
        x = torch.randn(8, 20)
        y = F.one_hot(torch.randint(0, 5, (8,)), 5).float()
        for _ in range(2):
            o.zero_grad(set_to_none=False)
            pnn.CrossEntropy()(m(x), y).sum().backward()
            o.step()
        states.append([p.detach().clone() for p in m.parameters()])
    for a, b in zip(*states):
        torch.testing.assert_close(a, b, rtol=2e-2, atol=2e-3)


# ── Parameters reach .grad as residuals on every route into the graph ──────────

ROUTES = {
    "mul view": lambda x, p: x * p.view(1, 3),
    "add unsqueeze": lambda x, p: x + p.unsqueeze(0),
    "add slice": lambda x, p: x[:, :2] + p[:2],
    "transposed": lambda x, p: x + p.view(3, 1).T,
    "view of view": lambda x, p: x * p.view(3, 1).view(1, 3),
    "sub method": lambda x, p: x.sub(p),
    "torch.sub": lambda x, p: torch.sub(x, p),
    "p.sub": lambda x, p: p.sub(x),
    "relu input": lambda x, p: pnn.ReLU()(p),
    "add_": lambda x, p: (x * 1.0).add_(p),
    "sub_": lambda x, p: (x * 1.0).sub_(p),
    "mul_": lambda x, p: (x * 1.0).mul_(p),
}


def _iadd(x, p):
    h = x * 1.0
    h += p
    return h


def _chained(x, p):  # the converted copy must not be converted again downstream
    return pnn.ReLU()(p).view(1, 3) + x


ROUTES["iadd"], ROUTES["module then op"] = _iadd, _chained


def _strided(p):  # non-contiguous, so reshape, flatten and contiguous must copy
    return p.view(1, 3).expand(2, 3).T


_IDX, _MASK = torch.tensor([0, 2, 2]), torch.tensor([True, False, True])

# Ops that only copy, select or rearrange: each is its output on a Parameter p, with
# x a constant here. The last entries are views, which need no patch of their own.
COPIES = {
    "cat": lambda x, p: torch.cat([p.view(1, 3), x]),
    "cat keyword": lambda x, p: torch.cat(tensors=[p.view(1, 3), x]),
    "concat": lambda x, p: torch.concat([p, x[0]]),
    "stack": lambda x, p: torch.stack([p, x[0]]),
    "hstack": lambda x, p: torch.hstack([p, x[0]]),
    "vstack": lambda x, p: torch.vstack([p, x[0]]),
    "tensor index": lambda x, p: p[_IDX],
    "list index": lambda x, p: p[[0, 1, 1]],
    "mask index": lambda x, p: p[_MASK],
    "clone": lambda x, p: p.clone(),
    "torch.clone": lambda x, p: torch.clone(p),
    "double": lambda x, p: p.double(),
    "half": lambda x, p: p.half(),
    "bfloat16": lambda x, p: p.bfloat16(),
    "to dtype": lambda x, p: p.to(torch.float64),
    "to keyword": lambda x, p: p.to(dtype=torch.float64),
    "type": lambda x, p: p.type(torch.float64),
    "type_as": lambda x, p: p.type_as(x.double()),
    "index_select": lambda x, p: p.index_select(0, _IDX),
    "torch.index_select": lambda x, p: torch.index_select(p, 0, _IDX),
    "gather": lambda x, p: p.gather(0, _IDX),
    "take": lambda x, p: p.take(_IDX),
    "take_along_dim": lambda x, p: p.take_along_dim(_IDX, 0),
    "where": lambda x, p: torch.where(_MASK, p, x[0]),
    "where scalar": lambda x, p: torch.where(_MASK, p, 0.0),
    "tensor where": lambda x, p: p.where(_MASK, x[0]),
    "masked_select": lambda x, p: p.masked_select(_MASK),
    "masked_fill": lambda x, p: p.masked_fill(_MASK, 0.0),
    "reshape copy": lambda x, p: _strided(p).reshape(-1),
    "flatten copy": lambda x, p: _strided(p).flatten(),
    "contiguous copy": lambda x, p: _strided(p).contiguous(),
    "repeat": lambda x, p: p.repeat(2),
    "tile": lambda x, p: p.tile(2),
    "repeat_interleave": lambda x, p: p.repeat_interleave(2),
    "flip": lambda x, p: p.flip(0),
    "roll": lambda x, p: p.roll(1),
    "F.pad": lambda x, p: F.pad(p, (1, 2)),
    "tril": lambda x, p: _strided(p).tril(),
    "expand then copy": lambda x, p: p.expand(4, 3).clone(),
    "slice then copy": lambda x, p: p[:2].clone(),
    "narrow then copy": lambda x, p: p.narrow(0, 1, 2).clone(),
    "split then copy": lambda x, p: p.split(2)[0].clone(),
    "chunk then copy": lambda x, p: p.chunk(3)[1].clone(),
}
# A copy chained into another op, which must not convert a second time. A copy selected
# from in part: the selection is what converts, or unselected coordinates would get p - 0.
CHAINS = {
    "copy of a copy": lambda x, p: p.clone().clone(),
    "cast of a copy": lambda x, p: p.clone().double(),
    "copy of a cast": lambda x, p: p.double().clone(),
    "cat of copies": lambda x, p: torch.cat([p.clone().view(1, 3), p.double().float().view(1, 3)]),
    "slice of a copy": lambda x, p: p.clone()[:2].clone(),
    "index of a cast": lambda x, p: p.double()[_IDX],
    "narrow of a cast": lambda x, p: p.double().narrow(0, 0, 2).clone(),
    "chunk of a copy": lambda x, p: p.clone().chunk(3)[0].clone(),
    "index of a transposed copy": lambda x, p: p.clone().view(3, 1).T[0, _IDX],
    "slice of a cat": lambda x, p: torch.cat([p, p])[:4].clone(),
    "index of a stack": lambda x, p: torch.stack([p, p])[0, _IDX],
}
ROUTES.update({f"{name} + x": (lambda f: lambda x, p: f(x, p).sum() + x)(f) for name, f in COPIES.items()})
ROUTES.update(COPIES)
ROUTES.update(CHAINS)
ROUTES.update({
    "explicit operand on a copy": lambda x, p: ops.parameter_operand(p.clone()),
    "explicit operand twice": lambda x, p: ops.parameter_operand(ops.parameter_operand(p)),
    "narrow": lambda x, p: x[:, :2] + p.narrow(0, 0, 2),
    "split": lambda x, p: x[:, :2] + p.split(2)[0],
    "chunk": lambda x, p: x[:, :1] + p.chunk(3)[1],
    "expand": lambda x, p: x + p.expand(4, 3),
    "copy into a sum": lambda x, p: p.clone().sum() + x,
    "slice of a copy into add": lambda x, p: x[:, :2] + p.clone()[:2],
    "cast into mul": lambda x, p: x.double() * p.double(),
    "copy into a module": lambda x, p: pnn.ReLU()(p.clone()[:2]),
    "module keyword input": lambda x, p: pnn.ReLU()(input=p),
    "copy add_": lambda x, p: p.clone().add_(x[0]),
    "copy mul_": lambda x, p: p.clone().mul_(x[0]),
    "copy sub": lambda x, p: p.clone().sub(x[0]),
})


@pytest.mark.parametrize("route", ROUTES)
def test_zero_misfit_leaves_a_parameter_alone(route):
    """Target == output: the residual is 0, so SGD(lr=1) must not move p."""
    p = tnn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    y = ROUTES[route](torch.ones(4, 3), p)
    y.backward(y.detach())
    torch.testing.assert_close(p.grad, torch.zeros(3))
    ProjectionSGD([p], lr=1.0).step()
    torch.testing.assert_close(p.detach(), torch.tensor([1.0, 2.0, 3.0]))


def test_parameter_moves_toward_its_target_through_a_view():
    p = tnn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    x = torch.zeros(4, 3)
    y = x + p.view(1, 3)
    y.backward(y.detach() + 1.0)
    assert (p.grad < 0).all()  # target above p: the residual is negative


def test_a_linear_input_parameter_gets_a_residual():
    torch.manual_seed(0)
    p = tnn.Parameter(torch.randn(5, 3))
    y = pnn.Linear(3, 2)(p)
    y.backward(y.detach())
    torch.testing.assert_close(p.grad, torch.zeros(5, 3), atol=1e-6, rtol=0)


def test_slice_leaves_the_rest_of_the_parameter_alone():
    p = tnn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    y = p[:2] + torch.ones(2)
    y.backward(y.detach() + 1.0)
    assert p.grad[2] == 0 and (p.grad[:2] != 0).all()


def _native_vjp(f, x, v, d):
    """d through f's plain autograd: gradient mode leaves these ops native."""
    config.update("use_projections", False)
    p = tnn.Parameter(v.clone())
    f(x, p).backward(d)
    config.update("use_projections", True)
    return p.grad


@pytest.mark.parametrize("route", {**COPIES, **CHAINS})
def test_a_misfit_reaches_the_parameter_as_a_residual(route):
    """Target y + d gives J^T (y - target) = -J^T d: signs, repeated coordinates,
    unselected coordinates and the cast back to p's dtype, against native autograd."""
    f = {**COPIES, **CHAINS}[route]
    torch.manual_seed(0)
    v, x = torch.tensor([1.0, 2.0, 3.0]), torch.ones(4, 3)
    p = tnn.Parameter(v.clone())
    y = f(x, p)
    d = torch.randn(y.shape).to(y.dtype)
    y.backward(y.detach() + d)
    assert p.grad.dtype == p.dtype
    tol = 2e-2 if y.dtype in (torch.half, torch.bfloat16) else 1e-6
    torch.testing.assert_close(p.grad, -_native_vjp(f, x, v, d), rtol=tol, atol=tol)


def test_repeated_indices_add_their_residuals():
    p = tnn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    y = p[torch.tensor([0, 0, 2])]
    y.backward(torch.tensor([2.0, 3.0, 3.0]))
    torch.testing.assert_close(p.grad, torch.tensor([-3.0, 0.0, 0.0]))


@pytest.mark.parametrize("select", [
    lambda p: p[torch.tensor([0, 1])],
    lambda p: p[torch.tensor([True, True, False])],
    lambda p: p.index_select(0, torch.tensor([0, 1])),
    lambda p: p.gather(0, torch.tensor([0, 1])),
    lambda p: p.take(torch.tensor([0, 1])),
    lambda p: p.clone()[:2],
    lambda p: p.double()[torch.tensor([0, 1])],
    lambda p: p.double().narrow(0, 0, 2),
    lambda p: p.clone().split(2)[0],
    lambda p: torch.cat([p, p])[:2],
])
def test_partial_selection_leaves_the_rest_at_zero(select):
    p = tnn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    y = select(p) + torch.ones(2, dtype=select(p).dtype)
    y.backward(y.detach() + 1.0)
    assert p.grad[2] == 0 and (p.grad[:2] < 0).all()


@pytest.mark.parametrize("op", [lambda p, a: torch.cat([p, a]), lambda p, a: torch.stack([p, a]),
                                lambda p, a: torch.where(_MASK, p, a)])
def test_an_activation_beside_a_parameter_keeps_its_target(op):
    torch.manual_seed(0)
    v, w = torch.tensor([1.0, 2.0, 3.0]), torch.tensor([5.0, 6.0, 7.0])
    p, a = tnn.Parameter(v.clone()), w.clone().requires_grad_()
    y = op(p, a)
    z = torch.randn(y.shape)
    y.backward(z)
    config.update("use_projections", False)
    q, b = v.clone().requires_grad_(), w.clone().requires_grad_()
    native = op(q, b)
    torch.testing.assert_close(a.grad, torch.autograd.grad(native, b, z, retain_graph=True)[0])
    torch.testing.assert_close(p.grad, torch.autograd.grad(native, q, y.detach() - z)[0])


def test_a_copy_can_be_written_in_place():
    p = tnn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    h = p.clone()
    h += torch.ones(3)
    h.backward(h.detach())
    torch.testing.assert_close(p.grad, torch.zeros(3))


def test_a_cast_forms_the_residual_in_its_dtype():
    p = tnn.Parameter(torch.tensor([1.0]))
    z = torch.tensor([1.0 + 1e-12], dtype=D)
    p.double().backward(z)
    assert p.grad.dtype == torch.float32 and p.grad.item() == pytest.approx(-1e-12, rel=1e-3)


def test_integer_and_string_results_pass_through():
    p = tnn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    assert p.to(torch.long).dtype == torch.long and not p.to(torch.long).requires_grad
    assert p.type() == "torch.FloatTensor"


def test_a_noop_cast_returns_the_parameter():
    p = tnn.Parameter(torch.ones(3))
    assert p.float() is p and p.to(torch.float32) is p and p.contiguous() is p and p.cpu() is p


def test_views_of_a_parameter_stay_views():
    p = tnn.Parameter(torch.ones(3))
    assert p[:2]._base is p and p.reshape(3, 1)._base is p and p.narrow(0, 0, 2)._base is p


def test_the_embedding_bridge_keeps_a_true_gradient():
    """nn.Embedding + Conversion hands F.embedding a gradient, not a target, so
    F.embedding stays native: a conversion would turn it back into the target."""
    torch.manual_seed(0)
    emb = tnn.Embedding(5, 3)
    ids = torch.tensor([0, 0, 4])
    h = emb(ids)
    z = torch.randn(3, 3)
    pxm.Conversion()(h).backward(z)
    g = torch.zeros(5, 3).index_add(0, ids, h.detach() - z)
    torch.testing.assert_close(emb.weight.grad, g)


@pytest.mark.parametrize("name", ["add_", "sub_", "mul_", "sub"])
def test_activations_keep_the_native_op(name):
    """Only a Parameter operand is rerouted: in-place residual adds on activations
    (e.g. `out += shortcut`) keep their existing behaviour."""
    a = torch.ones(3, requires_grad=True)
    b = torch.full((3,), 2.0, requires_grad=True)
    h = a * 1.0
    h = getattr(h, name)(b)
    assert "Projected" not in type(h.grad_fn).__name__


@pytest.mark.parametrize("route", COPIES)
def test_activations_keep_native_copies(route):
    """An activation in the Parameter's place: same graph as gradient mode, and its
    target arrives through plain autograd."""
    f, x, v = COPIES[route], torch.ones(4, 3), torch.tensor([1.0, 2.0, 3.0])
    a = v.clone().requires_grad_()
    y = f(x, a)
    y.backward(torch.ones_like(y))
    config.update("use_projections", False)
    b = v.clone().requires_grad_()
    native = f(x, b)
    native.backward(torch.ones_like(native))
    assert type(y.grad_fn) is type(native.grad_fn)
    torch.testing.assert_close(a.grad, b.grad)


def test_activation_in_place_on_a_slice_stays_native():
    h = torch.ones(4, 3, requires_grad=True) * 1.0
    view = h[:, :2]
    view += torch.ones(4, 2)
    assert "Projected" not in type(h.grad_fn).__name__


@pytest.mark.parametrize("route", COPIES)
def test_gradient_mode_keeps_native_copies(route):
    config.update("use_projections", False)
    p = tnn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    y = COPIES[route](torch.ones(4, 3), p)
    assert "ParameterOperand" not in type(y.grad_fn).__name__


@pytest.mark.parametrize("route", COPIES)
def test_no_grad_keeps_native_copies(route):
    p = tnn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    with torch.no_grad():
        y = COPIES[route](torch.ones(4, 3), p)
    assert y.grad_fn is None and not y.requires_grad


def test_module_casts_and_deepcopy_stay_parameters():
    m = pnn.Linear(3, 2).double()
    assert isinstance(m.weight, tnn.Parameter) and m.weight.dtype == D and m.weight.grad_fn is None
    assert copy.deepcopy(m).weight.grad_fn is None


def test_remove_overrides_restores_the_native_ops():
    from ptorch.core import overrides
    natives = dict(overrides._ORIG)
    p = tnn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    ptorch.remove_overrides()
    try:
        for (o, n), orig in natives.items():
            assert getattr(o, n) is orig
        y = torch.cat([p.view(1, 3), torch.ones(4, 3)])
        y.backward(y.detach())
        torch.testing.assert_close(p.grad, torch.tensor([1.0, 2.0, 3.0]))  # the raw target again
    finally:
        ptorch.apply_overrides()
    assert torch.cat is not natives[torch, "cat"]
    p.grad = None
    y = torch.cat([p.view(1, 3), torch.ones(4, 3)])
    y.backward(y.detach())
    torch.testing.assert_close(p.grad, torch.zeros(3))


def test_in_place_on_a_parameter_still_raises():
    p = tnn.Parameter(torch.ones(3))
    with pytest.raises(RuntimeError):
        p.add_(tnn.Parameter(torch.ones(3)))


@pytest.mark.parametrize("opt", [ProjectionSGD, ProjectionAdam, ProjectionMuon])
def test_optimizers_reject_plain_leaves(opt):
    with pytest.raises(TypeError, match="nn.Parameter"):
        opt([torch.ones(3, 3, requires_grad=True)], lr=1e-2)


def test_non_pairwise_solve_needs_a_weight():
    A, B = torch.randn(8, 6), torch.randn(6, 5, requires_grad=True)
    with pytest.raises(ValueError, match="pairwise"):
        ops.MatMulProjection.apply(A, B, 5, 1.0, 1.0, 1.0)
    W = tnn.Parameter(torch.randn(5, 6))
    with pytest.raises(ValueError, match="pairwise"):
        ops.MatMulProjection.apply(A, W.T, 5, 1.0, 1.0, 1.0, False, True)
