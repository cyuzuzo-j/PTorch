"""
Matmul Projection Sweep: α and g
=================================
For a single random (A, B) pair and a perturbed target Z_target,
call the bilinear projection sweeping both α and g, and record:

  • ||A_proj − A||  (input change)
  • ||B_proj − B||  (weight change)
  • ||z' − Z||      (output change — z' = <a'_j, b'_j> per element, the true projected output)
  • ||z' − Z_target|| (accuracy — how close the projection got to the target)

NOTE: z' is the true per-element inner product from the projection.
      A_proj @ B_proj ≠ z' because A_proj is an average over columns — it's
      a compromised vector that was never paired with B_proj in the projection.
      z' IS the correct "actual output" of the projection.

Produces:
  matmul_projection_sweep.png           – 1-D sweep comparing seq / iterative
  matmul_projection_sweep_combined.png  – combined 1-D overlay
  matmul_projection_sweep_2d.png        – 2-D heatmaps  (α × g)
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from ptorch.core.ops import matmul_proj_seq_pt, matmul_proj_iterative_pt
from functools import partial

# ─── Configuration ────────────────────────────────────────────────────────────
SEED       = 42
M, K, N    = 32, 64, 16          # A: (M,K)  B: (K,N)  Z: (M,N)
NOISE_STD  = 10.0                 # perturbation magnitude for Z_target

# 1-D sweep (α only, g = 1)
NUM_ALPHAS_1D = 80
ALPHA_MIN  = 1e-3
ALPHA_MAX  = 1e2

# 2-D sweep (α × g)
NUM_ALPHAS = 40
NUM_GS     = 40
G_MIN      = 1e-3
G_MAX      = 1e2

OUT_DIR = os.path.dirname(__file__)
OUT_FILE          = os.path.join(OUT_DIR, "matmul_projection_sweep.png")
OUT_FILE_COMBINED = os.path.join(OUT_DIR, "matmul_projection_sweep_combined.png")
OUT_FILE_2D       = os.path.join(OUT_DIR, "matmul_projection_sweep_2d.png")

# ─── Setup ────────────────────────────────────────────────────────────────────
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

A = torch.randn(M, K, device=device)
B = torch.randn(K, N, device=device)
Z = A @ B
Z_target = Z + NOISE_STD * torch.randn_like(Z)

target_perturbation = (Z_target - Z).norm().item()

print(f"Device : {device}")
print(f"A {tuple(A.shape)}, B {tuple(B.shape)}, Z {tuple(Z.shape)}")
print(f"||Z_target − Z|| = {target_perturbation:.4f}  (reference perturbation)\n")

# ═══════════════════════════════════════════════════════════════════════════════
#  1-D sweep over α  (g = 1)
# ═══════════════════════════════════════════════════════════════════════════════
alphas_1d = np.logspace(np.log10(ALPHA_MIN), np.log10(ALPHA_MAX), NUM_ALPHAS_1D)

results = {}
for mode, proj_fn in [
    ("seq (1 pass)",        matmul_proj_seq_pt),
    ("iterative (3 pass)",  partial(matmul_proj_iterative_pt, num_iters=3)),
    ("iterative (10 pass)", partial(matmul_proj_iterative_pt, num_iters=10)),
]:
    dA, dB, dZ, dZt = [], [], [], []
    print(f"\n── 1-D sweep [{mode}]: α ∈ [{ALPHA_MIN:.0e}, {ALPHA_MAX:.0e}], g = 1 ──")
    for i, alpha in enumerate(alphas_1d):
        A_proj, B_proj, Z_proj = proj_fn(A.clone(), B.clone(), Z_target.clone(),
                                          alpha=float(alpha))
        dA.append((A_proj - A).norm().item())
        dB.append((B_proj - B).norm().item())
        dZ.append((Z_proj - Z).norm().item())
        dZt.append((Z_proj - Z_target).norm().item())
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  α={alpha:>12.3e}  ||ΔA||={dA[-1]:>10.4f}  ||ΔB||={dB[-1]:>10.4f}"
                  f"  ||Δz'||={dZ[-1]:>10.4f}  ||z'−Zt||={dZt[-1]:>10.4f}")
    results[mode] = {"dA": dA, "dB": dB, "dZ": dZ, "dZt": dZt}

# ── 1-D Plot ──────────────────────────────────────────────────────────────────
modes = list(results.keys())
metrics = ["dA", "dB", "dZ", "dZt"]
metric_titles = [
    r"$\| A' - A \|_2$  (input change)",
    r"$\| B' - B \|_2$  (weight change)",
    r"$\| z' - Z \|_2$  (output change)",
    r"$\| z' - Z_t \|_2$  (accuracy)",
]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
n_modes = len(modes)

fig, axes = plt.subplots(n_modes, 4, figsize=(20, 4.5 * n_modes), sharex=True)

for ri, mode in enumerate(modes):
    for ci, (metric, title, color) in enumerate(zip(metrics, metric_titles, colors)):
        ax = axes[ri][ci]
        y = results[mode][metric]
        ax.plot(alphas_1d, y, linewidth=2, color=color, zorder=3)
        ax.fill_between(alphas_1d, y, alpha=0.12, color=color, zorder=2)
        ax.set_xscale("log")
        ax.set_ylabel("norm", fontsize=10)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        ax.axhline(0, color="k", linewidth=0.5)
        if ri == 0:
            ax.set_title(title, fontsize=11)
        if ri == n_modes - 1:
            ax.set_xlabel(r"$\alpha$", fontsize=12)
        if ci >= 2:
            ax.axhline(target_perturbation, color="gray", ls="--", lw=1.0,
                       label=r"$\|Z_t-Z\|$")
        if ci == 3:
            ax.axhline(0, color="green", ls=":", lw=1.0)
    axes[ri][0].annotate(mode, xy=(-0.35, 0.5), xycoords="axes fraction",
                          fontsize=11, fontweight="bold", rotation=90,
                          va="center", ha="center")

fig.suptitle(
    f"Projection variants  —  A ({M}×{K}), B ({K}×{N}), σ={NOISE_STD}, g=1\n"
    r"$z'_{{ij}} = \langle a'_i, b'_j \rangle$  (true per-element projected output)",
    fontsize=12, y=1.01,
)
plt.tight_layout()
plt.savefig(OUT_FILE, dpi=180, bbox_inches="tight")
print(f"\nSaved → {OUT_FILE}")

# ── 1-D combined overlay ─────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 4, figsize=(22, 5), sharex=True)

line_styles = ["-", "-.", ":"]
for ci, (metric, title, color) in enumerate(zip(metrics, metric_titles, colors)):
    ax = axes2[ci]
    for mode, ls in zip(modes, line_styles):
        ax.plot(alphas_1d, results[mode][metric], linewidth=2, ls=ls,
                color=color, label=mode, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\alpha$", fontsize=12)
    ax.set_ylabel("norm", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.legend(fontsize=8)
    if ci >= 2:
        ax.axhline(target_perturbation, color="gray", ls="--", lw=1.0)

fig2.suptitle(
    f"Projection variants overlay — A ({M}×{K}), B ({K}×{N}), σ={NOISE_STD}, g=1",
    fontsize=12, y=1.02,
)
plt.tight_layout()
plt.savefig(OUT_FILE_COMBINED, dpi=180, bbox_inches="tight")
print(f"Saved → {OUT_FILE_COMBINED}")

# ═══════════════════════════════════════════════════════════════════════════════
#  2-D sweep over (α, g)
# ═══════════════════════════════════════════════════════════════════════════════
alphas_2d = np.logspace(np.log10(ALPHA_MIN), np.log10(ALPHA_MAX), NUM_ALPHAS)
gs_2d     = np.logspace(np.log10(G_MIN),     np.log10(G_MAX),     NUM_GS)

# Grids: rows = g index, cols = alpha index
grid_dA  = np.zeros((NUM_GS, NUM_ALPHAS))
grid_dB  = np.zeros((NUM_GS, NUM_ALPHAS))
grid_dZ  = np.zeros((NUM_GS, NUM_ALPHAS))
grid_dZt = np.zeros((NUM_GS, NUM_ALPHAS))

total = NUM_ALPHAS * NUM_GS
print(f"\n── 2-D sweep: α ∈ [{ALPHA_MIN:.0e}, {ALPHA_MAX:.0e}] × g ∈ [{G_MIN:.0e}, {G_MAX:.0e}]"
      f"  ({NUM_ALPHAS}×{NUM_GS} = {total} pts) ──")

done = 0
for gi, g_val in enumerate(gs_2d):
    for ai, alpha_val in enumerate(alphas_2d):
        A_proj, B_proj, Z_proj = matmul_proj_seq_pt(
            A.clone(), B.clone(), Z_target.clone(),
            alpha=float(alpha_val), g=float(g_val),
        )
        grid_dA[gi, ai]  = (A_proj - A).norm().item()
        grid_dB[gi, ai]  = (B_proj - B).norm().item()
        grid_dZ[gi, ai]  = (Z_proj - Z).norm().item()
        grid_dZt[gi, ai] = (Z_proj - Z_target).norm().item()
        done += 1
    if (gi + 1) % 10 == 0 or gi == 0:
        print(f"  g={g_val:>10.3e}  row done  ({done}/{total})")

print(f"  2-D sweep complete.")

# ── 2-D Heatmap figure (2×2) ─────────────────────────────────────────────────
fig3, axes3 = plt.subplots(2, 2, figsize=(14, 11))

grids  = [grid_dA, grid_dB, grid_dZ, grid_dZt]
titles_2d = [
    r"$\| A' - A \|_2$  (input change)",
    r"$\| B' - B \|_2$  (weight change)",
    r"$\| z' - Z \|_2$  (output change)",
    r"$\| z' - Z_t \|_2$  (accuracy)",
]

for ax, grid, title in zip(axes3.flat, grids, titles_2d):
    grid_safe = np.where(grid > 0, grid, 1e-12)
    vmin, vmax = grid_safe[grid_safe > 1e-12].min(), grid_safe.max()

    im = ax.pcolormesh(
        alphas_2d, gs_2d, grid_safe,
        shading="nearest",
        norm=LogNorm(vmin=max(vmin, 1e-10), vmax=vmax),
        cmap="viridis",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\alpha$  (weight stiffness)", fontsize=11)
    ax.set_ylabel(r"$g$  (target stiffness)", fontsize=11)
    ax.set_title(title, fontsize=12)
    fig3.colorbar(im, ax=ax, pad=0.02)

fig3.suptitle(
    f"2-D projection sweep (sequential)  —  A ({M}×{K}), B ({K}×{N}), noise σ={NOISE_STD}\n"
    r"$z'_{ij} = \langle a'_i, b'_j \rangle$  (true per-element projected output)",
    fontsize=13, y=1.01,
)
plt.tight_layout()
plt.savefig(OUT_FILE_2D, dpi=180, bbox_inches="tight")
print(f"Saved → {OUT_FILE_2D}")
