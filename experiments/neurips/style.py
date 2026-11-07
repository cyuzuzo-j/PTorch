"""Shared figure style, taken from the paper's Figure 1 (fig:shallow-mlp), with
the paper's own KUL-ORANGE (main.tex) as the orange.

Every NeurIPS plot is made through this module so that all figures look alike:
- `apply_theme` sets one font size, line width, marker size and legend style;
- `subplots` gives every panel the same 7 x 5 in size (a two-panel figure is
  14 x 5), and `savefig` keeps that size instead of cropping to the contents,
  so all figures have the same aspect ratio per panel and, scaled to the same
  width per panel in LaTeX, the same printed font size;
- `color_map` colours a method the same in every figure: PTorch is always crimson,
  Torch orange, PJAX navy, PTorch l_inf lilac.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PANEL_SIZE = (7, 5)  # inches per panel
DPI = 300

NAVY = "#1f4e9c"        # PJAX
CRIMSON = "#c2185b"     # PTorch (l2)
GREEN = "#00a693"       # PTorch hybrid
ORANGE = "#e68c00"      # Torch; KUL-ORANGE, RGB(230, 140, 0)
BLUE = "#56b4e9"        # PTorch floor(0.3)
PURPLE = "#b57edc"      # PTorch l_inf
DARK_ORANGE = "#8a5400"  # PTorch floor(0.7); KUL-ORANGE!60!black

# Variant key (results CSV / YAML `labels:` key) -> colour.
VARIANT_COLORS = {
    "ptorch": CRIMSON, "ptorch_l2": CRIMSON, "ptorch_resnet8": CRIMSON, "ptorch_resnet6": CRIMSON, "l2": CRIMSON,
    "torch": ORANGE, "torch_resnet8": ORANGE, "torch_resnet6": ORANGE,
    "pjax": NAVY,
    "ptorch_floor0.3": BLUE,
    "ptorch_floor0.7": DARK_ORANGE,
    "ptorch_linf": PURPLE, "linf": PURPLE,
    "ptorch_hybrid": GREEN, "hybrid": GREEN,
    "init_he": CRIMSON, "init_normal": ORANGE, "init_orthogonal": NAVY,  # He normal: the PTorch default
}

# Colours for variants without a fixed one (optimizers, losses, ablation rows).
CYCLE = [NAVY, ORANGE, CRIMSON, BLUE, GREEN, PURPLE, DARK_ORANGE]


def color_map(variants):
    """Colour per variant: its fixed colour, else the next unused CYCLE colour."""
    colors = {v: VARIANT_COLORS[v] for v in variants if v in VARIANT_COLORS}
    free = [c for c in CYCLE if c not in colors.values()]
    for i, v in enumerate([v for v in variants if v not in colors]):
        colors[v] = free[i % len(free)] if free else CYCLE[i % len(CYCLE)]
    return colors


def sequential(n):
    """n colours along the navy -> teal -> green stretch of viridis, for ordered
    series such as network depth."""
    return [mpl.colormaps["viridis"](t) for t in np.linspace(0.25, 0.8, n)]


def apply_theme():
    sns.set_theme(style="whitegrid", context="paper", font_scale=2.0)
    mpl.rcParams.update({
        "lines.linewidth": 2,
        "lines.markersize": 8,
        "legend.fontsize": 12,
        "legend.title_fontsize": 12,
        "legend.frameon": False,
        "legend.loc": "lower right",
    })


def subplots(ncols=1):
    """A figure of `ncols` side-by-side panels, each PANEL_SIZE."""
    w, h = PANEL_SIZE
    return plt.subplots(1, ncols, figsize=(w * ncols, h))


def savefig(fig, path):
    """Save at the fixed figure size (no tight cropping) and close."""
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  saved {path}")
