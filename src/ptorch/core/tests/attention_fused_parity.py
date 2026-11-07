"""Parity tests for the fused causal-attention projection node.

Covers:
  - legacy five-node chain vs FusedCausalAttentionProjection over the config
    grid softmax {jacobian, rel_floor} x td_lambda {0, 0.5} x
    attn_value_adjoint {0, 1} x matmul_b_self_deflate {0, 1}:
    forward bit-identical, backward targets to fp32 tolerance;
  - full CausalSelfAttention module with config.fused_attn toggled.

Runs on CPU by design (the training GPU is usually occupied); pass
device="cuda" manually for a GPU spot-check.
"""

import torch
import torch._dynamo
import torch.nn.functional as F

# The grid below cycles alpha variants / b_self_deflate branches through
# matmul_proj; the default recompile limit (8) would silently flip later
# combos to eager kernels, whose ~ulp differences from inductor get amplified
# by the TD floor rescale and break tolerance depending on test order. Same
# fix as train_gpt_ptorch.py.
torch._dynamo.config.recompile_limit = 64

from ...config import config
from .. import ops
from ..attention_native import FusedCausalAttentionProjectionV2 as FusedCausalAttentionProjection
from ..ops import MatMulProjection, MaskedAddProjection, SoftmaxProjection
from ...nn.modules_experimental import GQAConsensusProjection

B, H, KV, T, HD = 2, 8, 4, 64, 16
OMEGA = HD ** 0.5


def _snapshot_config():
    return config.snapshot()


def _restore_config(snap):
    for key, val in snap.items():
        config.update(key, val)


def _set_campaign_config(softmax_mode, td_lambda, value_adjoint, b_deflate):
    config.update("use_projections", True)
    config.update("branch_mode", "delta_sum")
    config.update("projection_alpha_auto", True)
    config.update("softmax_target_mode", softmax_mode)
    config.update("td_lambda", td_lambda)
    config.update("td_mode", "norm")
    config.update("td_scope", "global")
    config.update("td_row", False)
    config.update("td_deflate", True)
    config.update("td_eps_lin", 1e-2)
    config.update("attn_value_adjoint", value_adjoint)
    config.update("matmul_b_self_deflate", b_deflate)
    config.update("use_muon_activations", False)


def _seed_td_trace(device, tensor_floor):
    """Mimic the state a loss backward leaves: g set (gate), norm-mode floor."""
    seed_norm = torch.tensor(3.0, device=device)
    ops.TD_TRACE.update(
        g=torch.randn(4, 4, device=device),
        seed_norm=seed_norm,
        floor=(seed_norm.double() if tensor_floor else float(seed_norm)),
        floor_row=None,
        seed_row=None,
    )


def _make_inputs(device, seed=0):
    torch.manual_seed(seed)
    q = (0.5 * torch.randn(B, H, T, HD, device=device)).requires_grad_()
    k = (0.5 * torch.randn(B, KV, T, HD, device=device)).requires_grad_()
    v = (0.5 * torch.randn(B, KV, T, HD, device=device)).requires_grad_()
    return q, k, v


def _legacy_chain(q, k, v, qk_match_g_to_omega=True):
    """Verbatim replica of CausalSelfAttention.forward's projection region
    (GQA expand -> QK solve -> mask add -> softmax -> PV solve)."""
    group = H // KV
    k_exp = GQAConsensusProjection.apply(k, group).reshape(B, H, T, HD)
    v_exp = GQAConsensusProjection.apply(v, group).reshape(B, H, T, HD)
    g = OMEGA if qk_match_g_to_omega else config.projection_g
    qk = MatMulProjection.apply(
        q, k_exp.transpose(-2, -1), config.attn_pairwise_num_steps,
        config.projection_alpha, g, OMEGA, True, False,
    )
    mask = torch.triu(
        torch.full((T, T), float("-inf"), device=q.device, dtype=qk.dtype),
        diagonal=1,
    )
    qk = MaskedAddProjection.apply(qk, mask[None, None, :, :])
    attn = SoftmaxProjection.apply(qk)
    y = MatMulProjection.apply(
        attn, v_exp, config.attn_pairwise_num_steps, config.projection_alpha,
        config.projection_g, 1.0, True, True,
    )
    return y


def _run(path, softmax_mode, td_lambda, value_adjoint, b_deflate,
         tensor_floor, device="cpu", seed=0):
    """One forward+backward; returns (y, q_target, k_target, v_target, floor)."""
    snap = _snapshot_config()
    try:
        _set_campaign_config(softmax_mode, td_lambda, value_adjoint, b_deflate)
        q, k, v = _make_inputs(device, seed)
        torch.manual_seed(100 + seed)
        _seed_td_trace(device, tensor_floor)
        if path == "fused":
            y = FusedCausalAttentionProjection.apply(q, k, v, OMEGA, OMEGA)
        else:
            y = _legacy_chain(q, k, v)
        z_y = y.detach() - 0.01 * torch.randn_like(y)   # a plausible target
        y.backward(z_y)
        floor = ops.TD_TRACE["floor"]
        floor = float(floor) if floor is not None else None
        return (y.detach(), q.grad.detach(), k.grad.detach(),
                v.grad.detach(), floor)
    finally:
        _restore_config(snap)
        ops.TD_TRACE.update(g=None, seed_norm=None, floor=None,
                            floor_row=None, seed_row=None)


def _assert_close(a, b, rtol, atol, what):
    if not torch.allclose(a, b, rtol=rtol, atol=atol):
        err = (a - b).abs().max()
        rel = err / b.abs().max().clamp_min(1e-12)
        raise AssertionError(f"{what}: max abs err {err:.3e}, rel {rel:.3e}")


def test_fused_matches_legacy_grid():
    for softmax_mode in ("jacobian", "rel_floor"):
        for td_lambda in (0.0, 0.5):
            for value_adjoint in (False, True):
                for b_deflate in (False, True):
                    tag = (f"softmax={softmax_mode} td={td_lambda} "
                           f"vadj={value_adjoint} bdefl={b_deflate}")
                    ref = _run("legacy", softmax_mode, td_lambda,
                               value_adjoint, b_deflate, tensor_floor=True)
                    got = _run("fused", softmax_mode, td_lambda,
                               value_adjoint, b_deflate, tensor_floor=True)
                    assert torch.equal(ref[0], got[0]), f"forward differs: {tag}"
                    # fp32 bar: the fused path recomputes the pattern as
                    # exp(qk - lse) (~1 ulp from the softmax kernel) and may
                    # amplify it through the TD rescale; real bugs are >=1e-2.
                    for r, n, name in zip(ref[1:4], got[1:4], ("q", "k", "v")):
                        _assert_close(n, r, 1e-4, 2e-5, f"{name} target ({tag})")
                    if td_lambda != 0.0:
                        assert abs(ref[4] - got[4]) <= 1e-6 * abs(ref[4]), \
                            f"floor differs: {tag} {ref[4]} vs {got[4]}"


def test_full_module_dispatch():
    """CausalSelfAttention end to end with config.fused_attn toggled: input
    target and projection-weight targets must agree."""
    from ...nn.modules_experimental import CausalSelfAttention

    snap = _snapshot_config()
    try:
        _set_campaign_config("jacobian", 0.5, True, False)
        torch.manual_seed(7)
        dim = H * HD
        attn = CausalSelfAttention(dim, H, KV, rope_base=10000.0,
                                   qk_gain_init=1.5, norm="l2",
                                   qk_match_g_to_omega=True)
        for m in attn.modules():
            if hasattr(m, "use_cache"):
                m.use_cache = False
        x0 = torch.randn(B, T, dim)
        results = {}
        for flag in ("0", "1"):
            config.update("fused_attn", flag == "1")
            attn.zero_grad(set_to_none=True)
            _seed_td_trace(x0.device, tensor_floor=True)
            x = x0.clone().requires_grad_()
            y = attn(x)
            torch.manual_seed(8)
            y.backward(y.detach() - 0.01 * torch.randn_like(y))
            results[flag] = (
                x.grad.detach().clone(),
                {n: p.grad.detach().clone() for n, p in attn.named_parameters()
                 if p.grad is not None},
            )
        # Looser than the raw-chain grid: the module wraps the fused region in
        # rotary/norm/Linear nodes that accumulate the ~1-ulp pattern-recompute
        # difference (measured rel ~2e-6); real bugs show up at >=1e-2.
        _assert_close(results["1"][0], results["0"][0], 1e-4, 2e-5, "module x target")
        for name, ref in results["0"][1].items():
            _assert_close(results["1"][1][name], ref, 1e-4, 2e-5,
                          f"module param target {name}")
    finally:
        _restore_config(snap)
        ops.TD_TRACE.update(g=None, seed_norm=None, floor=None,
                            floor_row=None, seed_row=None)


if __name__ == "__main__":
    test_fused_matches_legacy_grid()
    print("fused vs legacy grid: OK")
    test_full_module_dispatch()
    print("full module dispatch: OK")
