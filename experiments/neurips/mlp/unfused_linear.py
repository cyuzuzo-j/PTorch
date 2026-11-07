"""Ablation row R1 ("plain cyclic projections"): the l2 Linear projection done
the unfused way, as in PJAX. Every (sample m, output n) pair gets its own copy
of the input row and the weight column, projected independently onto
{a·b = z}; the copies are then averaged (consensus). This materializes
(M, N, K) tensors, the O(N_in N_out B) memory the fused solve
(ptorch.core.ops.matmul_proj) avoids.

It uses the same scalar solve as matmul_proj (one safeguarded Newton step
from t = 0 per pair, steps clamped to 0.5) and forms the weight residual the
same way (no target subtraction), so its input targets and weight residuals
equal the fused ones up to rounding: R1 and R2 differ only in cost.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ptorch.config import config


def _pairwise_projection(A, B, Z, alpha=1.0, g=1.0, num_steps=1):
    """A (M, K), B (K, N), Z (M, N) -> consensus target for A and consensus
    residual B - B_bar for B."""
    a = A[:, None, :]                     # (M, 1, K)
    b = B.T[None, :, :]                   # (1, N, K)
    p = (a * b).sum(-1)                   # (M, N), via the (M, N, K) product
    q = (a * a).sum(-1) + alpha * (b * b).sum(-1)   # (M, N) by broadcasting

    eps, max_step = 1e-5, 0.5
    t_max = math.sqrt(max(alpha - eps, eps))
    t = torch.zeros_like(p)
    lo, hi = torch.full_like(p, -t_max), torch.full_like(p, t_max)
    for _ in range(num_steps):
        t2 = t * t
        d = alpha - t2
        num = alpha * (p * (alpha + t2) + t * q)
        f = num / d ** 2 - Z + t / g ** 2
        fp = (alpha * (2 * t * p + q) * d + 4 * t * num) / d ** 3 + 1 / g ** 2
        lo = torch.where(f < 0, t, lo)
        hi = torch.where(f > 0, t, hi)
        t_new = t - (f / fp.clamp_min(1e-12)).clamp(-max_step, max_step)
        t_new = torch.where((t_new < lo) | (t_new > hi), 0.5 * (lo + hi), t_new)
        t = torch.where((f.abs() < 1e-10) | (t_new == t), t, t_new)

    d = (alpha - t * t)[..., None]        # (M, N, 1)
    tt = t[..., None]
    a_prop = alpha * (a + tt * b) / d     # (M, N, K): one proposal per output
    # b - (alpha b + t a) / d, with alpha/d - 1 = t^2/d as in the fused solve.
    b_res = (tt * tt / d) * b + (tt / d) * a   # (M, N, K), negated below
    return a_prop.mean(1), -b_res.mean(0).T


class _UnfusedMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight_t, bias, alpha, g):
        ctx.save_for_backward(x, weight_t, bias)
        ctx.alpha, ctx.g = alpha, g
        return torch.addmm(bias, x, weight_t)

    @staticmethod
    def backward(ctx, z_target):
        x, w_t, bias = ctx.saved_tensors
        A = F.pad(x, (0, 1), value=1.0)
        B = torch.cat([w_t, bias])
        A_proj, B_res = _pairwise_projection(A, B, z_target, ctx.alpha, ctx.g)
        if config.mm_prox_mu > 0.0 and config.mm_prox_scope in ("all", "other"):
            # PTORCH_MM_PROX_MU: the fused path's A-side override (ops.MatMulProjection.backward).
            G = (B @ B.T).double()
            mu = config.mm_prox_mu * G.shape[0] / G.diagonal().sum().clamp_min(1e-12)
            W = (mu * torch.linalg.solve(torch.eye(len(G), dtype=G.dtype, device=G.device) + mu * G, B.double()))
            dA = (z_target - A @ B) @ W.to(A.dtype).T
            A_proj = A + (dA * ((A_proj - A).norm() / dA.norm().clamp_min(1e-30)) if config.mm_prox_dirswap else dA)
        return A_proj[:, :-1], B_res[:-1], B_res[-1:], None, None


class UnfusedLinear(nn.Module):
    """Drop-in for ptorch.nn.Linear(norm='l2') with the unfused solve.
    Parameters are created and initialized in the same order as Linear, so a
    given seed gives the same initial weights."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(1, out_features))
        nn.init.kaiming_normal_(self.weight)

    def forward(self, x):
        return _UnfusedMatMul.apply(x, self.weight.T, self.bias, config.projection_alpha, config.projection_g)
