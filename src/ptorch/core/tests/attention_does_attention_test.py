"""Functional test: does the dot-product + sparsemax projection backward
actually DO attention?

Motivated by the 2026-07-09 consensus audit ("the consensus on the dot product
calculates nothing / averages random values"). These tests check the isolated
attention core — QK card -> causal mask -> sparsemax card -> PV card, with
q/k/v as free leaves (no Linears, no rotary, no norms, no TD) — for the three
properties that together mean "it does attention":

  (i)   test_reachable_target_descent: iterating x <- x̄ (the projection
        targets) drives ||y - z_y|| down by >3x within 100 iterations for a
        reachable target, in both the legacy composition and fused V2.
  (ii)  test_redirects_attention_within_support: with only q free and a target
        demanding the value of a non-argmax support key j*, the chain shifts
        pattern mass onto j* (argmax flips, P[j*] grows, residual shrinks) —
        the dot-product backward transports real attention-routing signal.
  (iii) test_backward_direction_matches_gradient: the one-step target
        displacement (x - x̄) is positively aligned with the true autograd
        gradient of 0.5*||y - z_y||^2 for q, k and v — i.e. the consensus
        average is NOT averaging random values into nothing.

Plus one characterization of the known failure mode, so it is never again
mistaken for a consensus bug:

  (iv)  test_face_lock_matches_gradient_dead_zone: when the demanded key is
        OFF the sparsemax support (score below threshold), the graph card
        correctly refuses the face change, the QK card receives an identity
        target and returns q̄ == q exactly — and the TRUE autograd gradient is
        exactly zero there too (sparsemax dead zone). Frozen retrieval in this
        regime is inherent to sparsemax, shared with backprop, not a solver
        defect. (Empirically GD is worse: it saturates onto a wrong key and
        locks at maximal loss; the projection chain at least stays put.)

CPU-only, T=8, seconds to run. Thresholds carry ~2x margin over measured
values (descent ratio 0.15 @100 it; retrieval P[j*] 0.376->0.640 @400 it;
cos q/k/v ~0.5/0.7/0.9).
"""
import torch
import torch._dynamo

torch._dynamo.config.recompile_limit = 64

from ...config import config
from ..ops import MaskedAddProjection
from ..attention_native import (
    FusedCausalAttentionProjectionV2,
    SparsemaxProjection,
    sparsemax_forward,
)
from .attention_native_test import _restore_config, _snapshot_config

B, H, T, HD = 1, 1, 8, 8
OMEGA = HD ** 0.5


def _set_core_config():
    config.update("use_projections", True)
    config.update("projection_alpha_auto", True)
    config.update("attention_activation", "sparsemax")
    config.update("sparsemax_max_face_iters", 2)
    config.update("td_lambda", 0.0)
    config.update("use_muon_activations", False)
    config.update("attn_value_adjoint", False)
    config.update("attn_exact_transport", "off")
    config.update("attn_strip_self", False)
    config.update("matmul_b_self_deflate", False)


def _mask(device=None, dtype=None):
    return torch.triu(
        torch.full((T, T), float("-inf"), device=device, dtype=dtype), 1
    )[None, None]


def _forward_legacy(q, k, v):
    from ...nn.attention_native import pairwise_matmul

    qk = pairwise_matmul(q, k.transpose(-2, -1), omega=OMEGA, g=OMEGA)
    qk = MaskedAddProjection.apply(qk, _mask(q.device, qk.dtype))
    P = SparsemaxProjection.apply(qk)
    return pairwise_matmul(P, v, value_solve=True), P


def _forward_fused(q, k, v):
    y = FusedCausalAttentionProjectionV2.apply(q, k, v, OMEGA, OMEGA, None)
    with torch.no_grad():
        vis = torch.ones(T, T, dtype=torch.bool, device=q.device).tril_()
        P, _ = sparsemax_forward((q @ k.transpose(-2, -1)) / OMEGA,
                                 vis[None, None])
    return y, P


def _iterate(fwd, tensors, free, z_y, n_iters):
    """Projection fixed-point iteration: replace each free leaf by its target
    (the backward 'grad' IS the target x̄ in this framework)."""
    cur = dict(tensors)
    for _ in range(n_iters):
        leaves = {n: cur[n].detach().clone().requires_grad_(n in free)
                  for n in cur}
        y, P = fwd(leaves["q"], leaves["k"], leaves["v"])
        y.backward(z_y)
        with torch.no_grad():
            for n in free:
                cur[n] = leaves[n].grad.detach().clone()
    leaves = {n: cur[n].detach().clone().requires_grad_(True) for n in cur}
    y, P = fwd(leaves["q"], leaves["k"], leaves["v"])
    return y.detach(), P.detach()


# ── (i) reachable target: fixed-point iteration descends the residual ────────

def test_reachable_target_descent():
    snap = _snapshot_config()
    try:
        _set_core_config()
        for name, fwd in (("legacy", _forward_legacy), ("fused", _forward_fused)):
            torch.manual_seed(0)
            q, k, v = (torch.randn(B, H, T, HD) for _ in range(3))
            qs, ks, vs = (t + 0.3 * torch.randn_like(t) for t in (q, k, v))
            with torch.no_grad():
                z_y, _ = fwd(qs, ks, vs)
            z_y = z_y.detach()
            with torch.no_grad():
                y0, _ = fwd(q, k, v)
            r0 = (y0 - z_y).norm().item()
            y_end, _ = _iterate(fwd, dict(q=q, k=k, v=v), ("q", "k", "v"),
                                z_y, 100)
            ratio = (y_end - z_y).norm().item() / r0
            print(f"descent[{name}]: ||y - z_y|| ratio after 100 it = {ratio:.3f}")
            assert ratio < 0.35, \
                f"{name}: targets do not descend the output residual " \
                f"(ratio {ratio:.3f}, expected < 0.35)"
    finally:
        _restore_config(snap)


# ── (ii) q-only retrieval within the support: mass moves to the right key ────

def test_redirects_attention_within_support():
    snap = _snapshot_config()
    try:
        _set_core_config()
        torch.manual_seed(1)
        q = 0.5 * torch.randn(B, H, T, HD)
        k = 0.5 * torch.randn(B, H, T, HD)
        v = 3.0 * torch.eye(T)[None, None]  # distinct payload per position
        with torch.no_grad():
            y0, P0 = _forward_legacy(q, k, v)
        row = P0[0, 0, -1]
        support = (row > 0).nonzero().flatten().tolist()
        assert len(support) >= 2, "test setup: need a multi-key support"
        jstar = next(j for j in support if j != int(row.argmax()))
        p_init = float(row[jstar])
        # demand the last query retrieve payload jstar (currently non-argmax)
        z_y = y0.detach().clone()
        z_y[0, 0, -1] = v[0, 0, jstar]
        r0 = (y0[0, 0, -1] - z_y[0, 0, -1]).norm().item()
        y_end, P_end = _iterate(_forward_legacy, dict(q=q, k=k, v=v), ("q",),
                                z_y, 400)
        p_end = float(P_end[0, 0, -1, jstar])
        ratio = (y_end[0, 0, -1] - z_y[0, 0, -1]).norm().item() / r0
        print(f"retrieval: P[j*] {p_init:.3f} -> {p_end:.3f}, "
              f"argmax {int(row.argmax())} -> {int(P_end[0, 0, -1].argmax())}, "
              f"residual ratio {ratio:.3f}")
        assert int(P_end[0, 0, -1].argmax()) == jstar, \
            "pattern argmax did not move to the demanded key"
        assert p_end > p_init + 0.15, \
            f"P[j*] barely moved: {p_init:.3f} -> {p_end:.3f}"
        assert ratio < 0.75, f"retrieval residual did not shrink: {ratio:.3f}"
    finally:
        _restore_config(snap)


# ── (iii) one-step targets align with the true gradient (not random) ─────────

def test_backward_direction_matches_gradient():
    snap = _snapshot_config()
    try:
        _set_core_config()
        vis = torch.ones(T, T, dtype=torch.bool).tril_()[None, None]
        cos_acc = {"q": [], "k": [], "v": []}
        for seed in range(5):
            torch.manual_seed(seed)
            base = {n: torch.randn(B, H, T, HD) for n in ("q", "k", "v")}
            leaves = {n: t.clone().requires_grad_() for n, t in base.items()}
            y, _ = _forward_legacy(leaves["q"], leaves["k"], leaves["v"])
            z_y = y.detach() - 0.05 * torch.randn_like(y)
            y.backward(z_y)
            # autograd reference on a differentiable replica of the chain
            ref = {n: t.clone().requires_grad_() for n, t in base.items()}
            config.update("use_projections", False)
            P, _ = sparsemax_forward(
                (ref["q"] @ ref["k"].transpose(-2, -1)) / OMEGA, vis)
            loss = 0.5 * ((P @ ref["v"] - z_y) ** 2).sum()
            loss.backward()
            config.update("use_projections", True)
            for n in ("q", "k", "v"):
                upd = (leaves[n].detach() - leaves[n].grad.detach()).flatten()
                grad = ref[n].grad.detach().flatten()
                cos_acc[n].append(float(torch.nn.functional.cosine_similarity(
                    upd, grad, dim=0)))
        means = {n: sum(c) / len(c) for n, c in cos_acc.items()}
        print("cos((x - x̄), grad):",
              {n: f"{m:+.3f}" for n, m in means.items()})
        for n, floor in (("q", 0.25), ("k", 0.40), ("v", 0.60)):
            assert means[n] > floor, \
                f"{n} target direction decorrelated from gradient: " \
                f"mean cos {means[n]:.3f} (floor {floor})"
            assert min(cos_acc[n]) > 0.0, \
                f"{n} target anti-aligned on a seed: {cos_acc[n]}"
    finally:
        _restore_config(snap)


# ── (iv) off-support demand: frozen — and so is the true gradient ─────────────

def test_face_lock_matches_gradient_dead_zone():
    snap = _snapshot_config()
    try:
        _set_core_config()
        torch.manual_seed(1)
        q = torch.randn(B, H, T, HD)
        k = 2.0 * torch.randn(B, H, T, HD)   # saturated one-hot pattern
        v = 3.0 * torch.eye(T)[None, None]
        with torch.no_grad():
            y0, P0 = _forward_legacy(q, k, v)
        row = P0[0, 0, -1]
        jstar = next(j for j in range(T) if float(row[j]) == 0.0)
        z_y = y0.detach().clone()
        z_y[0, 0, -1] = v[0, 0, jstar]       # demands an off-support key
        ql = q.clone().requires_grad_()
        y, _ = _forward_legacy(ql, k, v)
        y.backward(z_y)
        dq = (ql.grad.detach() - q).norm() / q.norm()
        # true gradient of the same objective is exactly zero (dead zone)
        qg = q.clone().requires_grad_()
        vis = torch.ones(T, T, dtype=torch.bool).tril_()[None, None]
        config.update("use_projections", False)
        P, _ = sparsemax_forward((qg @ k.transpose(-2, -1)) / OMEGA, vis)
        loss = 0.5 * ((P @ v - z_y) ** 2).sum()
        loss.backward()
        gnorm = float(qg.grad.norm())
        print(f"dead zone: rel ||q̄ - q|| = {float(dq):.2e}, "
              f"||autograd grad|| = {gnorm:.2e}")
        assert float(dq) < 1e-5, \
            f"chain moved q in the sparsemax dead zone: rel {float(dq):.2e}"
        assert gnorm == 0.0, \
            f"expected exact-zero gradient in the dead zone, got {gnorm:.2e}"
    finally:
        _restore_config(snap)


# ── (v) sparsemax_gamma unlocks the face-lock (the dead-zone fix) ─────────────

def _offsupport_retrieval(fwd, sparsemax_gamma, n_iters=400):
    """Off-support retrieval instance (gap ~0.2): frozen at default γ,
    expected to flip at γ=4. Returns (P[j*], argmax_hit, residual_ratio)."""
    config.update("sparsemax_gamma", sparsemax_gamma)
    torch.manual_seed(1)
    q = 0.5 * torch.randn(B, H, T, HD)
    k = 0.5 * torch.randn(B, H, T, HD)
    v = 3.0 * torch.eye(T)[None, None]
    with torch.no_grad():
        y0, P0 = fwd(q, k, v)
    row = P0[0, 0, -1]
    jstar = next(j for j in range(T) if float(row[j]) == 0.0)
    z_y = y0.detach().clone()
    z_y[0, 0, -1] = v[0, 0, jstar]
    r0 = (y0[0, 0, -1] - z_y[0, 0, -1]).norm().item()
    y_end, P_end = _iterate(fwd, dict(q=q, k=k, v=v), ("q",), z_y, n_iters)
    ratio = (y_end[0, 0, -1] - z_y[0, 0, -1]).norm().item() / r0
    return (float(P_end[0, 0, -1, jstar]),
            int(P_end[0, 0, -1].argmax()) == jstar, ratio)


def test_sparsemax_gamma_unlocks_face_lock():
    """The dead-zone fix: config.sparsemax_gamma raises the activation card's
    target stiffness so demanded face changes become affordable. Default
    (None -> projection_g) must stay frozen on the off-support instance
    (documents the lock); gamma=4 must flip it — in both the legacy
    composition and fused V2. Measured: default res ratio ~1.01;
    gamma=4 P[j*]~0.56, flip @~212, res ~0.46."""
    snap = _snapshot_config()
    try:
        _set_core_config()
        for name, fwd in (("legacy", _forward_legacy), ("fused", _forward_fused)):
            p_frozen, hit_frozen, ratio_frozen = _offsupport_retrieval(fwd, None)
            p_fix, hit_fix, ratio_fix = _offsupport_retrieval(fwd, 4.0)
            print(f"face-unlock[{name}]: default P*={p_frozen:.2f} "
                  f"res={ratio_frozen:.2f} | gamma=4 P*={p_fix:.2f} "
                  f"flip={hit_fix} res={ratio_fix:.2f}")
            assert not hit_frozen and ratio_frozen > 0.9, \
                f"{name}: off-support instance no longer frozen at default " \
                f"gamma — the fix's premise changed (res {ratio_frozen:.2f})"
            assert hit_fix, f"{name}: gamma=4 did not flip the pattern to j*"
            assert p_fix > 0.4, f"{name}: gamma=4 P[j*] too small: {p_fix:.2f}"
            assert ratio_fix < 0.6, \
                f"{name}: gamma=4 residual did not drop: {ratio_fix:.2f}"
    finally:
        _restore_config(snap)


if __name__ == "__main__":
    test_reachable_target_descent()
    print("reachable-target descent: OK")
    test_redirects_attention_within_support()
    print("within-support retrieval: OK")
    test_backward_direction_matches_gradient()
    print("target-vs-gradient alignment: OK")
    test_face_lock_matches_gradient_dead_zone()
    print("face-lock == gradient dead zone: OK")
    test_sparsemax_gamma_unlocks_face_lock()
    print("sparsemax_gamma face unlock: OK")
