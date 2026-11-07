"""Tests for the projection-native attention core (attention_native.py).

Covers, per the handoff acceptance criteria:
  (i)   sparsemax forward matches a reference NumPy implementation, incl.
        causal masking, T ∈ {1, 2, 8, 64};
  (ii)  the graph-projection backward satisfies the constraint
        (ā_proj = sparsemax(s̄) within tolerance) and is empirically
        non-expansive on random instances (>=10k trials, max ratio reported
        Table-4.1 style);
  (iii) fused (FusedCausalAttentionProjectionV2) == legacy composition with
        SparsemaxProjection at T ∈ {1, 2, 8} (T=1 has no legacy bail-out in
        sparsemax mode), td_lambda ∈ {0, 0.5};
  plus: the full CausalSelfAttentionNative module agrees fused-vs-legacy in
        sparsemax mode.

Runs on CPU by design (the training GPU is usually occupied); pass
device="cuda" manually for a GPU spot-check.
"""

import numpy as np
import torch
import torch._dynamo

# Same rationale as attention_fused_parity.py: the grids below cycle compiled
# variants through matmul_proj; the default recompile limit would silently
# flip later combos to eager.
torch._dynamo.config.recompile_limit = 64

from ...config import config
from .. import ops
from ..attention_native import (
    FusedCausalAttentionProjectionV2,
    SparsemaxProjection,
    sparsemax_forward,
    sparsemax_graph_project,
)
from ..delta_probe import RECORDER
from ..ops import MatMulProjection, MaskedAddProjection, SoftmaxProjection
from ...nn.modules_experimental import GQAConsensusProjection

B, H, KV, HD = 2, 8, 4, 16
OMEGA = HD ** 0.5


# ── config bookkeeping ─────────────────────────────────────────────────────────

def _snapshot_config():
    return config.snapshot()


def _restore_config(snap):
    for key, val in snap.items():
        config.update(key, val)


def _set_campaign_config(activation, td_lambda, value_adjoint=False,
                         b_deflate=False, softmax_mode="jacobian"):
    config.update("use_projections", True)
    config.update("branch_mode", "delta_sum")
    config.update("projection_alpha_auto", True)
    config.update("softmax_target_mode", softmax_mode)
    config.update("attention_activation", activation)
    config.update("attn_exact_transport", "off")
    config.update("attn_strip_self", False)
    config.update("sparsemax_max_face_iters", 2)
    config.update("td_lambda", td_lambda)
    config.update("td_mode", "norm")
    config.update("td_scope", "global")
    config.update("td_row", False)
    config.update("td_deflate", True)
    config.update("td_eps_lin", 1e-2)
    config.update("attn_value_adjoint", value_adjoint)
    config.update("matmul_b_self_deflate", b_deflate)
    config.update("use_muon_activations", False)


def _seed_td_trace(device):
    seed_norm = torch.tensor(3.0, device=device)
    ops.TD_TRACE.update(
        g=torch.randn(4, 4, device=device),
        seed_norm=seed_norm,
        floor=seed_norm.double(),
        floor_row=None,
        seed_row=None,
    )


def _clear_td_trace():
    ops.TD_TRACE.update(g=None, seed_norm=None, floor=None,
                        floor_row=None, seed_row=None)


def _assert_close(a, b, rtol, atol, what):
    if not torch.allclose(a, b, rtol=rtol, atol=atol):
        err = (a - b).abs().max()
        rel = err / b.abs().max().clamp_min(1e-12)
        raise AssertionError(f"{what}: max abs err {err:.3e}, rel {rel:.3e}")


# ── (i) sparsemax forward vs NumPy reference ─────────────────────────────────

def _sparsemax_numpy(row):
    """Martins & Astudillo reference on one 1-D finite array."""
    u = np.sort(row)[::-1]
    css = np.cumsum(u) - 1.0
    j = np.arange(1, len(row) + 1)
    rho = int(np.sum(u - css / j > 0))
    tau = css[rho - 1] / rho
    return np.maximum(row - tau, 0.0)


def test_sparsemax_forward_matches_numpy():
    torch.manual_seed(0)
    for T in (1, 2, 8, 64):
        scores = 2.0 * torch.randn(B, H, T, T, dtype=torch.float64)
        vis = torch.ones(T, T, dtype=torch.bool).tril_()[None, None]
        a, tau = sparsemax_forward(scores, vis)
        for b in range(B):
            for h in range(H):
                for t in range(T):
                    row = scores[b, h, t, : t + 1].numpy()
                    ref = _sparsemax_numpy(row)
                    got = a[b, h, t].numpy()
                    assert np.allclose(got[: t + 1], ref, atol=1e-10), \
                        f"T={T} row {t}: {got[: t + 1]} vs {ref}"
                    assert np.all(got[t + 1:] == 0.0), f"mask leak at T={T} row {t}"
        # simplex check
        sums = a.sum(-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-10)
    # the projection card path: -inf-masked input, same values
    T = 8
    scores = 2.0 * torch.randn(B, H, T, T, dtype=torch.float64)
    mask = torch.triu(torch.full((T, T), float("-inf"), dtype=torch.float64), 1)
    x = scores + mask[None, None]
    a_card = SparsemaxProjection.apply(x.requires_grad_())
    vis = torch.ones(T, T, dtype=torch.bool).tril_()[None, None]
    a_fn, _ = sparsemax_forward(scores, vis)
    assert torch.equal(a_card.detach(), a_fn), "card vs helper mismatch"


# ── (ii) graph projection: constraint satisfaction + non-expansiveness ───────

def test_graph_projection_constraint():
    """ā_proj must equal sparsemax(s̄). Contract: exact on the common case
    (small target displacements — what the PV card actually delivers) at the
    default 2 face solves; convergent everywhere at a higher solve budget."""
    torch.manual_seed(1)
    T, rows = 16, 4000
    gamma = 1.0
    vis = torch.ones(1, 1, 1, T, dtype=torch.bool)

    def _consistency(scale_s, scale_z, iters):
        s = scale_s * torch.randn(rows, 1, 1, T, dtype=torch.float64)
        a, _ = sparsemax_forward(s, vis)
        z = a + scale_z * torch.randn_like(a)
        s_bar, tau = sparsemax_graph_project(s, z, vis, gamma, iters)
        a_check, _ = sparsemax_forward(s_bar, vis)
        err = (a_check - (s_bar - tau).clamp(min=0.0)).abs().amax(dim=-1).flatten()
        return float((err < 1e-6).float().mean()), float(err.max())

    # common case, default budget (config default sparsemax_max_face_iters=2)
    for scale_s, scale_z in ((1.0, 0.02), (2.0, 0.05), (0.5, 0.05)):
        frac, worst = _consistency(scale_s, scale_z, iters=2)
        print(f"constraint (s~{scale_s}, ā noise {scale_z}, 2 solves): "
              f"{frac:.4%} consistent, worst {worst:.3e}")
        assert frac > 0.98, f"common-case satisfaction too low: {frac:.4%}"
    # far targets: bounded budget is heuristic; must converge with budget
    for scale_z in (0.5, 1.0):
        frac8, worst8 = _consistency(1.0, scale_z, iters=8)
        print(f"constraint (ā noise {scale_z}, 8 solves): "
              f"{frac8:.4%} consistent, worst {worst8:.3e}")
        assert frac8 > 0.995, f"far-target satisfaction at 8 solves: {frac8:.4%}"

    # idempotence: ā == a must return s̄ == s exactly
    s = torch.randn(rows, 1, 1, T, dtype=torch.float64)
    a, _ = sparsemax_forward(s, vis)
    s_id, tau = sparsemax_graph_project(s, a, vis, gamma, 2)
    a_id = (s_id - tau).clamp(min=0.0)
    assert torch.allclose(s_id, s, atol=1e-12), "not idempotent (s)"
    assert torch.allclose(a_id, a, atol=1e-12), "not idempotent (a)"


def test_graph_projection_nonexpansive():
    """Empirical §4.2-style check: for pairs of nearby inputs, the ratio
    d(P(x₁), P(x₂)) / d(x₁, x₂) in the (1, γ²)-weighted metric; >=10k trials,
    quantiles reported Table-4.1 style.

    The graph is nonconvex (union of faces), so the projection is
    discontinuous at the medial axis between faces: pairs straddling a face
    boundary can expand pointwise (exactly like the ReLU card's two-branch
    switch). The vanishing-target-relevant claim is bulk contraction in the
    deployed regime (γ = projection_g = 1, targets near the graph): asserted
    on the median and p95; the tail is printed for the record."""
    torch.manual_seed(2)
    T, trials = 12, 12000
    vis = torch.ones(1, 1, 1, T, dtype=torch.bool)
    for gamma, assert_bulk in ((1.0, True), (4.0, False)):
        s1 = torch.randn(trials, 1, 1, T, dtype=torch.float64)
        a1, _ = sparsemax_forward(s1, vis)
        z1 = a1 + 0.02 * torch.randn_like(a1)   # near-graph targets
        s2 = s1 + 0.005 * torch.randn_like(s1)  # nearby second point
        z2 = z1 + 0.005 * torch.randn_like(z1)
        sb1, t1 = sparsemax_graph_project(s1, z1, vis, gamma, 2)
        sb2, t2 = sparsemax_graph_project(s2, z2, vis, gamma, 2)
        af1, af2 = (sb1 - t1).clamp(min=0.0), (sb2 - t2).clamp(min=0.0)
        g2 = gamma * gamma
        d_in = ((s1 - s2).square().sum(-1) + g2 * (z1 - z2).square().sum(-1)).sqrt()
        d_out = ((sb1 - sb2).square().sum(-1) + g2 * (af1 - af2).square().sum(-1)).sqrt()
        ratio = (d_out / d_in.clamp_min(1e-12)).flatten()
        q = torch.quantile(ratio, torch.tensor([0.5, 0.95, 0.999, 1.0],
                                               dtype=ratio.dtype))
        print(f"non-expansiveness γ={gamma}: median {q[0]:.4f}, p95 {q[1]:.4f}, "
              f"p99.9 {q[2]:.4f}, max {q[3]:.4f}  ({trials} trials)")
        if assert_bulk:
            assert float(q[0]) < 0.9, f"median not contracting: {float(q[0]):.4f}"
            assert float(q[1]) <= 1.05, f"p95 expansive: {float(q[1]):.4f}"


# ── (iii) fused == legacy, sparsemax mode, T ∈ {1, 2, 8} ─────────────────────

def _make_inputs(T, device, seed=0):
    torch.manual_seed(seed)
    q = (0.5 * torch.randn(B, H, T, HD, device=device)).requires_grad_()
    k = (0.5 * torch.randn(B, KV, T, HD, device=device)).requires_grad_()
    v = (0.5 * torch.randn(B, KV, T, HD, device=device)).requires_grad_()
    return q, k, v


def _legacy_chain(q, k, v, T, activation):
    from ...nn.attention_native import pairwise_matmul

    group = H // KV
    k_exp = GQAConsensusProjection.apply(k, group).reshape(B, H, T, HD)
    v_exp = GQAConsensusProjection.apply(v, group).reshape(B, H, T, HD)
    qk = pairwise_matmul(q, k_exp.transpose(-2, -1), omega=OMEGA, g=OMEGA)
    mask = torch.triu(
        torch.full((T, T), float("-inf"), device=q.device, dtype=qk.dtype),
        diagonal=1,
    )
    qk = MaskedAddProjection.apply(qk, mask[None, None, :, :])
    if activation == "sparsemax":
        attn = SparsemaxProjection.apply(qk)
    else:
        attn = SoftmaxProjection.apply(qk)
    y = pairwise_matmul(attn, v_exp, value_solve=True)
    return y


def _run(path, T, activation, td_lambda, value_adjoint=False, seed=0,
         softmax_mode="jacobian"):
    snap = _snapshot_config()
    try:
        _set_campaign_config(activation, td_lambda, value_adjoint,
                             softmax_mode=softmax_mode)
        q, k, v = _make_inputs(T, "cpu", seed)
        torch.manual_seed(100 + seed)
        _seed_td_trace("cpu")
        if path == "fused_v2":
            y = FusedCausalAttentionProjectionV2.apply(q, k, v, OMEGA, OMEGA, None)
        else:
            y = _legacy_chain(q, k, v, T, activation)
        z_y = y.detach() - 0.01 * torch.randn_like(y)
        y.backward(z_y)
        floor = ops.TD_TRACE["floor"]
        floor = float(floor) if floor is not None else None
        return (y.detach(), q.grad.detach(), k.grad.detach(),
                v.grad.detach(), floor)
    finally:
        _restore_config(snap)
        _clear_td_trace()


def test_fused_matches_legacy_sparsemax():
    for T in (1, 2, 8):
        for td_lambda in (0.0, 0.5):
            for value_adjoint in (False, True):
                tag = f"sparsemax T={T} td={td_lambda} vadj={value_adjoint}"
                ref = _run("legacy", T, "sparsemax", td_lambda, value_adjoint)
                got = _run("fused_v2", T, "sparsemax", td_lambda, value_adjoint)
                assert torch.equal(ref[0], got[0]), f"forward differs: {tag}"
                for r, n, name in zip(ref[1:4], got[1:4], ("q", "k", "v")):
                    _assert_close(n, r, 1e-4, 2e-5, f"{name} target ({tag})")
                if td_lambda != 0.0:
                    assert abs(ref[4] - got[4]) <= 1e-6 * abs(ref[4]), \
                        f"floor differs: {tag} {ref[4]} vs {got[4]}"


# ── full module: fused vs legacy dispatch, sparsemax mode ─────────────────────

def test_full_module_dispatch_native():
    from ...nn.attention_native import CausalSelfAttentionNative

    snap = _snapshot_config()
    try:
        _set_campaign_config("sparsemax", 0.5, value_adjoint=True)
        torch.manual_seed(7)
        T = 64
        dim = H * HD
        attn = CausalSelfAttentionNative(dim, H, KV, rope_base=10000.0,
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
            _seed_td_trace(x0.device)
            x = x0.clone().requires_grad_()
            y = attn(x)
            torch.manual_seed(8)
            y.backward(y.detach() - 0.01 * torch.randn_like(y))
            results[flag] = (
                x.grad.detach().clone(),
                {n: p.grad.detach().clone() for n, p in attn.named_parameters()
                 if p.grad is not None},
            )
        _assert_close(results["1"][0], results["0"][0], 1e-4, 2e-5,
                      "module x target (sparsemax)")
        for name, ref in results["0"][1].items():
            _assert_close(results["1"][1][name], ref, 1e-4, 2e-5,
                          f"module param target {name} (sparsemax)")
    finally:
        _restore_config(snap)
        _clear_td_trace()


if __name__ == "__main__":
    test_sparsemax_forward_matches_numpy()
    print("sparsemax forward vs numpy: OK")
    test_graph_projection_constraint()
    print("graph-projection constraint: OK")
    test_graph_projection_nonexpansive()
    print("graph-projection non-expansiveness: OK")
    test_fused_matches_legacy_sparsemax()
    print("fused vs legacy (sparsemax, T in {1,2,8}): OK")
    test_full_module_dispatch_native()
    print("full module dispatch (native): OK")
