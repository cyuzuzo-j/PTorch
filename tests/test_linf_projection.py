"""The l_inf bilinear solve (ops.matmul_proj_linf) against a brute-force root of
F(eps) = sum_k M_k(eps) + eps/g^2 - |z - h.theta| (paper, l_inf derivation)."""
import pytest
import torch

from ptorch.core import ops


def _reach(h, th, eps, increase):
    """Max change of h.theta in the requested direction with |dh_k|, |dth_k| <= eps."""
    same, opp = (h + th).abs(), (h - th).abs()
    a, b = (same, opp) if increase else (opp, same)
    return torch.maximum(a * eps + eps ** 2, b * eps - eps ** 2).sum()


def _bisect_radius(h, th, z, g):
    d = z - h @ th
    target, increase = d.abs(), bool(d >= 0)
    lo, hi = 0.0, float(target) * g * g
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _reach(h, th, mid, increase) + mid / (g * g) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@pytest.mark.parametrize("g", [1.0, 0.5, 2.0])
def test_linf_radius_matches_bruteforce(g):
    torch.manual_seed(0)
    K = 7
    for _ in range(20):
        h, th = torch.randn(K, dtype=torch.float64), torch.randn(K, dtype=torch.float64)
        z = torch.randn((), dtype=torch.float64) * 3
        A, B, Z = h[None, :], th[:, None], z.reshape(1, 1)
        A_p, B_p, Z_p, eps = ops.matmul_proj_linf(A, B, Z, g=g, num_steps=30)
        ref = _bisect_radius(h, th, z, g)
        assert abs(float(eps) - ref) < 1e-8
        # single pair: no consensus averaging, so the point lies on the constraint
        assert abs(float(A_p @ B_p - Z_p)) < 1e-8
        # and it moved every coordinate by at most the radius
        assert (A_p - A).abs().max() <= ref + 1e-9 and (B_p - B).abs().max() <= ref + 1e-9


def test_linf_one_step_is_first_order_radius():
    """num_steps=1 keeps the old behavior: one Newton step from eps = 0."""
    torch.manual_seed(1)
    A, B, Z = torch.randn(4, 5, dtype=torch.float64), torch.randn(5, 3, dtype=torch.float64), \
        torch.randn(4, 3, dtype=torch.float64)
    *_, eps = ops.matmul_proj_linf(A, B, Z, num_steps=1)
    S = torch.sign(Z - A @ B)
    u = (A[:, :, None] + B[None]).abs()
    v = (A[:, :, None] - B[None]).abs()
    slope0 = torch.where(S[:, None, :] == 1, torch.maximum(u, v), torch.maximum(v, u)).sum(1)
    expected = (Z - A @ B).abs() / (slope0 + 1.0 + 1e-5)
    assert torch.allclose(eps, expected)


def test_linf_more_steps_reduce_residual():
    torch.manual_seed(2)
    A, B = torch.randn(1, 16, dtype=torch.float64), torch.randn(16, 1, dtype=torch.float64)
    Z = A @ B + 5.0
    res = []
    for n in (1, 5):
        A_p, B_p, Z_p, _ = ops.matmul_proj_linf(A, B, Z, num_steps=n)
        res.append(float((A_p @ B_p - Z_p).abs()))
    assert res[1] < 1e-6 < res[0]


def test_linf_sign_of_zero_is_one():
    """h_k + w_k = 0 and h_k - w_k = 0 (h_k = w_k = 0): the coordinate still takes a corner."""
    A = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    B = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    Z = torch.tensor([[3.0]], dtype=torch.float64)
    A_p, B_p, _, eps = ops.matmul_proj_linf(A, B, Z, num_steps=30)
    assert float(eps) > 0
    assert torch.allclose(A_p[0, 0], eps.reshape(())) and torch.allclose(B_p[0, 0], eps.reshape(()))
