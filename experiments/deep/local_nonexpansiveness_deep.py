"""
Local Non-Expansiveness Test — Deep Linear MLP
==============================================
Empirically tests Assumption (Local Non-Expansiveness) on the *actual*
(P_1, P_2) pairs produced by the forward + backward passes of an L-layer
linear MLP trained on the identity task with the projection framework.

For layer i+1 at a recorded training step we capture
    P_1 = (h^{(i)},     theta^{(i+1)}, h^{(i+1)})       (forward)
    P_2 = (h^{(i)},     theta^{(i+1)}, h_bar^{(i+1)})   (backward target)

Since P_1 already lies on the constraint set A_{i+1}, Pi(P_1) = P_1 and the
assumption reduces to
    ||Pi_{A_{i+1}}(P_2) - P_1||_2  <=  ||P_2 - P_1||_2.

The RHS is just ||h_bar^{(i+1)} - h^{(i+1)}||_2 (the only component that
differs). The LHS is computed via `matmul_proj`, the exact metric projection
solver, with many Newton iterations in float64 for accuracy.

Output: one PDF + PNG with three panels (per-layer histogram, summary curves
vs. distance-from-output for every depth, per-depth violation rate) and a
console summary.
"""

import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn

from ptorch.core.ops import MSEProjection, matmul_proj
from ptorch.nn.modules import Linear
from ptorch.optim_static import ProjectionSGD


sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)


D = 32
BATCH_SIZE = 16
N_SEEDS = 3
N_STEPS = 1000
DEPTHS = [2, 4, 8, 16, 32]
RECORD_STEPS = [1, N_STEPS // 2, N_STEPS]

PROJECTION_NEWTON_STEPS = 200          # high-accuracy metric projection
PROJECTION_DTYPE = torch.float64       # double precision for the solve
VIOLATION_TOL = 1e-5                   # absorb Newton residual

# Two filters on ||P_2 - P_1||:
#   - MIN_DENOM: drop records where den is exactly numerical noise.
#   - VALID_DENOM: ratio is only meaningful well above the Newton-residual
#     floor of matmul_proj (~1e-7..1e-8 in float64). Records with
#     den < VALID_DENOM are reported as "degenerate" (target signal has
#     vanished, cf. vanishing_target.py) and excluded from the ratio
#     statistics / plots; they are not violations of the assumption.
MIN_DENOM = 1e-12
VALID_DENOM = 1e-6


class IdentityMLP(nn.Module):
    """Linear MLP with hooks that capture the non-expansiveness ratio per
    layer per recorded step.

    Forward hook: capture A = h^{(i)}, B = theta^{(i+1)}, Z_fwd = h^{(i+1)}.
    Backward output-hook (registered only on recorded steps): receives the
    gradient on the layer's output, which under the projection framework is
    exactly h_bar^{(i+1)} (see `MatMulProjection.backward` in core/ops.py).
    Inside the hook we run `matmul_proj` on (A, B, h_bar) and compare to the
    forward triple.
    """

    def __init__(self, depth, d=D):
        super().__init__()
        self.depth = depth
        self.d = d
        self.layers = nn.ModuleList([Linear(d, d, bias=False) for _ in range(depth)])
        self.record_now = False
        self.records = []  # list of dicts: layer_idx, from_out, ratio, num, den
        self._register_hooks()

    def _register_hooks(self):
        for idx, layer in enumerate(self.layers):
            from_out = self.depth - 1 - idx

            def make_hook(layer_idx, from_output, lyr):
                def fwd_hook(module, inp, out):
                    if not self.record_now:
                        return
                    if not out.requires_grad:
                        return

                    A = inp[0].detach().clone()
                    B = lyr.weight.detach().t().clone()
                    Z_fwd = out.detach().clone()
                    alpha = lyr.alpha
                    g = lyr.g
                    omega = lyr.omega

                    def bwd_target_hook(grad):
                        Z_target = grad.detach().clone()

                        A_hi = A.to(PROJECTION_DTYPE).contiguous()
                        B_hi = B.to(PROJECTION_DTYPE).contiguous()
                        Z_fwd_s = (Z_fwd.to(PROJECTION_DTYPE) * omega).contiguous()
                        Z_tgt_s = (Z_target.to(PROJECTION_DTYPE) * omega).contiguous()

                        den = (Z_tgt_s - Z_fwd_s).norm(p=2).item()
                        if den < MIN_DENOM:
                            return

                        A_p, B_p, Z_p, _ = matmul_proj(
                            A_hi.clone(), B_hi.clone(), Z_tgt_s.clone(),
                            alpha=alpha, g=g, omega=omega,
                            num_steps=PROJECTION_NEWTON_STEPS,
                        )

                        num_sq = (
                            ((A_p - A_hi) ** 2).sum()
                            + ((B_p - B_hi) ** 2).sum()
                            + ((Z_p - Z_fwd_s) ** 2).sum()
                        )
                        num = num_sq.sqrt().item()
                        ratio = num / den

                        self.records.append({
                            'layer_idx': layer_idx,
                            'from_out': from_output,
                            'ratio': ratio,
                            'num': num,
                            'den': den,
                        })

                    out.register_hook(bwd_target_hook)
                return fwd_hook
            layer.register_forward_hook(make_hook(idx, from_out, layer))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def run_one(depth, seed, device):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = IdentityMLP(depth=depth, d=D).to(device)
    optimizer = ProjectionSGD(model.parameters())

    for step in range(1, N_STEPS + 1):
        model.record_now = step in RECORD_STEPS

        x = torch.randn(BATCH_SIZE, D, device=device)
        y = x.clone()

        model.train()
        optimizer.zero_grad()
        preds = model(x)
        loss = MSEProjection.apply(preds, y)
        loss.backward()
        optimizer.step()

    return model.records


def run_experiment(device):
    all_records = {d: [] for d in DEPTHS}
    for depth in DEPTHS:
        print(f"\n--- Depth L={depth} ---")
        for seed in range(N_SEEDS):
            recs = run_one(depth, seed, device)
            all_records[depth].extend(recs)
            print(f"  seed {seed + 1}/{N_SEEDS}: {len(recs)} records")
    return all_records


def _valid_arrays(recs):
    """Return (ratio, from_out) arrays restricted to records where the
    target signal is large enough that the ratio is meaningful."""
    r = np.array([rec['ratio'] for rec in recs])
    fo = np.array([rec['from_out'] for rec in recs])
    den = np.array([rec['den'] for rec in recs])
    mask = den >= VALID_DENOM
    return r[mask], fo[mask], mask.sum(), len(recs)


def summarize(all_records):
    print("\n================ Per-depth summary ================")
    print("(records with ||P_2-P_1|| < {:.0e} excluded as vanishing-signal degenerate.)".format(VALID_DENOM))
    print(f"{'depth':>5}  {'n_valid':>8} {'n_deg':>6}  {'max r':>8}  {'p95 r':>8}  {'mean r':>8}  {'%>1':>6}  | output-half: {'n':>5} {'max':>8}  {'%>1':>6}")
    for depth in DEPTHS:
        recs = all_records[depth]
        if not recs:
            print(f"{depth:>5}  no records")
            continue
        r, fo, n_valid, n_all = _valid_arrays(recs)
        n_deg = n_all - n_valid
        if n_valid == 0:
            print(f"{depth:>5}  {n_valid:>8d} {n_deg:>6d}  all records degenerate")
            continue

        half = fo < max(1, depth // 2)
        max_r = r.max()
        p95_r = np.percentile(r, 95)
        mean_r = r.mean()
        viol = 100.0 * np.mean(r > 1.0 + VIOLATION_TOL)

        if half.any():
            r_h = r[half]
            n_h = int(half.sum())
            max_h = r_h.max()
            viol_h = 100.0 * np.mean(r_h > 1.0 + VIOLATION_TOL)
        else:
            n_h, max_h, viol_h = 0, float('nan'), float('nan')

        print(f"{depth:>5}  {n_valid:>8d} {n_deg:>6d}  {max_r:>8.4f}  {p95_r:>8.4f}  {mean_r:>8.4f}  {viol:>5.1f}%  |             {n_h:>5d} {max_h:>8.4f}  {viol_h:>5.1f}%")


def plot_results(all_records, out_dir="images"):
    """Two-panel plot mirroring `local_nonexpansiveness_matmul.ipynb` style,
    with depth L on the role that the neighbourhood scale eps played there.

    Each depth contributes one pooled distribution of ratios (across all
    layers, seeds, and recorded steps), with vanishing-signal records
    excluded via VALID_DENOM.
    """
    out_dir = os.path.join(os.path.dirname(__file__), out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Pool valid ratios per depth
    ratios_by_depth = {}
    for depth in DEPTHS:
        recs = all_records[depth]
        rs = [r['ratio'] for r in recs if r['den'] >= VALID_DENOM]
        if rs:
            ratios_by_depth[depth] = np.asarray(rs)

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    # ---- max / p95 / mean vs depth ----
    palette = sns.color_palette("viridis", 3)

    depths_with_data = [d for d in DEPTHS if d in ratios_by_depth]
    max_r = [ratios_by_depth[d].max() for d in depths_with_data]
    p95_r = [np.percentile(ratios_by_depth[d], 95) for d in depths_with_data]
    mean_r = [ratios_by_depth[d].mean() for d in depths_with_data]

    ax.plot(depths_with_data, max_r,  '^-',  label='max',         color=palette[0])
    ax.plot(depths_with_data, p95_r,  's--', label='95th pctile', color=palette[1])
    ax.plot(depths_with_data, mean_r, 'o-',  label='mean',        color=palette[2])
    ax.axhline(1.0, color='gray', ls='--', lw=1.5, label='threshold')

    ax.set_xscale('log', base=2)
    ax.set_xticks(depths_with_data)
    ax.set_xticklabels([str(d) for d in depths_with_data])
    ax.set_xlabel('Depth L')
    ax.set_ylabel(r'$\|\Pi_{A_{i+1}}(P_2)-P_1\| \,/\, \|P_2-P_1\|$')
    ax.set_title('Non-Expansiveness vs Network Depth')
    ax.legend()

    plt.tight_layout()
    out_pdf = os.path.join(out_dir, "local_nonexpansiveness_deep.pdf")
    out_png = os.path.join(out_dir, "local_nonexpansiveness_deep.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"\nSaved figure to {out_pdf}")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    records = run_experiment(device)
    summarize(records)
    plot_results(records)
