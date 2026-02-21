"""Shared helpers for profiling scripts."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

# Ensure pjax is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from pjax.core.computation import (
    Array as ArrayNode,
    Operation,
    Parameter as ParamNode,
    ShapeTransform,
)
from pjax.optim import get_graph

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)


def save_graph(loss_fn, params, filename, title, figsize=(12, 8), font_size=6):
    """Build the computation graph from *loss_fn* and save a PNG.

    Args:
        loss_fn:   callable ``params -> Computation`` (the symbolic loss).
        params:    FrozenDict of parameter values.
        filename:  output PNG filename (relative to outputs/).
        title:     plot title.
        figsize:   matplotlib figure size.
        font_size: label font size.

    Returns:
        The networkx DiGraph so callers can inspect it.
    """
    # wrap concrete values as symbolic Parameter nodes
    p = {n: ParamNode(v, name=n) for n, v in params.items()}
    comp = loss_fn(p)
    g = get_graph(comp)

    colors = [
        "#4CAF50" if isinstance(n, ParamNode) else
        "#2196F3" if isinstance(n, ArrayNode) else
        "#FF9800" if isinstance(n, Operation) else
        "#9E9E9E"
        for n in g.nodes
    ]

    pos = nx.spring_layout(g, iterations=500, seed=0)
    plt.figure(figsize=figsize)
    nx.draw(g, pos, with_labels=True, node_color=colors,
            node_size=300, font_size=font_size, arrows=True)
    plt.title(title)
    path = os.path.join(OUT, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Graph  → {path}  ({g.number_of_nodes()} nodes, {g.number_of_edges()} edges)")
    return g
