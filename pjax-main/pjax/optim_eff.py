"""Efficient projection-based optimizers that follow backpropagation order.

Instead of building a bipartite graph via ``networkx`` and detecting bipartite
sets, these optimizers partition the computation graph into BFS layers using a
lightweight topological sort and walk them output → inputs (like back-prop).

Key advantages over the ``optim.py`` bipartite approach:

* **No networkx dependency** – graph construction and partitioning use plain
  Python dicts/sets built from an iterative DFS topological sort.
* **Works on any DAG** – no bipartite requirement; the BFS-layer partitioning
  generalises naturally (like ``CyclicOptimizer`` but cheaper).
* **Same projection semantics** – uses ``multiple_projection_eff`` which is
  functionally identical to ``optim.multiple_projection``.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import Sequence

import jax
from jax import numpy as jnp

from .core.computation import Array, Computation, Operation, Parameter, ShapeTransform
from .core.frozen_dict import FrozenDict, freeze


# ────────────────────────────────────────────────────────
#  Lightweight graph utilities (no networkx)
# ────────────────────────────────────────────────────────

def _topo_sort(root: Computation):
    """Return nodes in topological order (inputs first) via iterative DFS."""
    visited = set()
    order = []
    stack = [(root, False)]
    while stack:
        node, processed = stack.pop()
        if processed:
            if node not in visited:
                visited.add(node)
                order.append(node)
            continue
        if node in visited:
            continue
        stack.append((node, True))
        for parent in reversed(node.parents):
            if parent not in visited:
                stack.append((parent, False))
    return order


def _children_map(topo_order):
    """Build  node → [child, …]  from topological ordering."""
    children = {n: [] for n in topo_order}
    for node in topo_order:
        for parent in node.parents:
            if parent in children:
                children[parent].append(node)
    return children


def _pruned_children_map(topo_order, children_of):
    """Build a children map that skips ``ShapeTransform``s.

    Parents of a ``ShapeTransform`` are connected directly to its children,
    equivalent to ``optim.prune_shape_transforms`` but without networkx.
    """
    pruned: dict[Computation, list[Computation]] = {
        n: [] for n in topo_order if not isinstance(n, ShapeTransform)
    }
    for node in topo_order:
        if isinstance(node, ShapeTransform):
            continue
        for child in children_of[node]:
            # walk through chains of ShapeTransforms
            frontier = [child]
            while frontier:
                c = frontier.pop()
                if isinstance(c, ShapeTransform):
                    frontier.extend(children_of[c])
                else:
                    pruned[node].append(c)
    return pruned


def _out_degree(node, pruned_children):
    """Number of non-ShapeTransform children."""
    return len(pruned_children.get(node, []))


# ────────────────────────────────────────────────────────
#  BFS-layer partitioning (output → inputs, like CyclicOptimizer)
# ────────────────────────────────────────────────────────

def _bfs_partitions(topo_order, children_of):
    """Partition non-ShapeTransform nodes into BFS layers starting from outputs.

    Returns a list of sets, first set = output nodes, last set = leaf
    inputs.  This mirrors ``CyclicOptimizer._partition`` but without
    networkx.
    """
    pruned_children = _pruned_children_map(topo_order, children_of)

    # Build a reverse adjacency (pruned predecessors)
    pruned_preds: dict[Computation, set[Computation]] = {
        n: set() for n in pruned_children
    }
    for node, kids in pruned_children.items():
        for kid in kids:
            pruned_preds[kid].add(node)

    # First partition: all output (sink) nodes
    partitions = [
        set(n for n in pruned_children if _out_degree(n, pruned_children) == 0)
    ]

    # BFS backwards until we've covered all nodes
    visited = set(partitions[0])
    while len(visited) < len(pruned_children):
        layer = set()
        for node in partitions[-1]:
            for pred in pruned_preds[node]:
                if pred not in visited:
                    layer.add(pred)
        if not layer:
            break
        partitions.append(layer)
        visited |= layer

    return partitions


# ────────────────────────────────────────────────────────
#  Core helpers (same semantics as optim.py, no networkx)
# ────────────────────────────────────────────────────────

def _get_value(computation, inputs):
    """Evaluate a single node given current *inputs*."""
    if isinstance(computation, Array):
        return computation.value
    if isinstance(computation, Parameter):
        return inputs[computation][0]
    if isinstance(computation, Operation):
        return computation.operation(*inputs[computation])
    if isinstance(computation, ShapeTransform):
        return computation.transform(
            *[_get_value(p, inputs) for p in computation.parents]
        )


def _gather_and_average_outputs(node, inputs, children_of):
    """Average the feedback from all children of *node*."""
    kids = children_of[node]
    if not kids:
        return None
    vals = []
    for child in kids:
        vals.extend(_get_childs_input(child, node, inputs, children_of))
    stacked = jnp.stack(vals)
    return jnp.sum(stacked, axis=0) / len(vals)


def _get_childs_input(child, parent, inputs, children_of):
    """Get the value(s) that *child* receives from *parent*."""
    idxs = [i for i, p in enumerate(child.parents) if p == parent]

    if isinstance(child, Operation):
        return [inputs[child][i] for i in idxs]

    if isinstance(child, ShapeTransform):
        transform_inputs = [_get_value(p, inputs) for p in child.parents]
        output = _gather_and_average_outputs(child, inputs, children_of)
        values = child.inverse(*transform_inputs, output)
        if isinstance(values, jnp.ndarray):
            return [values]
        return [values[i] for i in idxs]

    raise ValueError(f"Unexpected child type {type(child)}")


def _projection(computation, inputs, children_of):
    """Project a single node (same as ``optim.projection``)."""
    if isinstance(computation, Array):
        return inputs
    output = _gather_and_average_outputs(computation, inputs, children_of)
    projected = computation.projection(*inputs[computation], output)
    return inputs.set(computation, list(projected))


def multiple_projection_eff(
    inputs: FrozenDict,
    partition: set,
    children_of: dict,
    pruned_children: dict,
):
    """Project every node in *partition*, then refresh children inputs.

    Functionally identical to ``optim.multiple_projection`` but uses
    pre-computed dicts instead of a networkx graph.
    """
    for computation in partition:
        inputs = _projection(computation, inputs, children_of)

    # Refresh children (using the pruned map to skip ShapeTransforms)
    children_to_update = set(
        child
        for comp in partition
        for child in pruned_children.get(comp, [])
    )
    for computation in children_to_update:
        if isinstance(computation, Operation):
            inputs = inputs.set(
                computation,
                [_get_value(p, inputs).astype(jnp.bfloat16) for p in computation.parents],
            )
    return inputs


# ────────────────────────────────────────────────────────
#  Efficient Optimizer base class
# ────────────────────────────────────────────────────────

class EfficientOptimizer(ABC):
    """Projection-based optimizer using BFS-layer partitions (no networkx).

    The graph is partitioned into BFS layers (output → inputs) and each
    layer is projected in sequence, exactly like ``CyclicOptimizer`` but
    without the ``networkx`` overhead.

    Sub-classes only need to implement ``_step``.

    Args:
        steps_per_update: projection iterations per ``update`` call.
        change_projection_order: reverse the layer ordering if ``True``.
    """

    def __init__(self, steps_per_update: int = 50, change_projection_order: bool = False):
        self.steps_per_update = steps_per_update
        self.change_projection_order = change_projection_order
        self.uses_p = False
        self.uses_q = False
        self.add_projection_at_end = False
        # Cache for graph topology (identical across training steps)
        self._graph_cache = None

    def update(self, fun, params: FrozenDict, steps_per_update=None):
        steps_per_update = steps_per_update or self.steps_per_update

        # 1. Build computation graph (plain Python, no networkx)
        params = {name: Parameter(value, name=name) for name, value in params.items()}
        computation = fun(params)

        # 2. Reuse cached graph topology or compute it once
        topo_order = _topo_sort(computation)
        children_of = _children_map(topo_order)
        pruned_children = _pruned_children_map(topo_order, children_of)
        partitions = _bfs_partitions(topo_order, children_of)
        self._graph_cache = (topo_order, children_of, pruned_children, partitions)
        # 3. Initialise inputs dict
        inputs = {}
        for node in topo_order:
            if isinstance(node, Parameter):
                inputs[node] = [node.value.astype(jnp.bfloat16)]
            elif isinstance(node, Operation):
                inputs[node] = [
                    p.value.astype(jnp.bfloat16) for p in node.parents
                ]
        inputs = freeze(inputs)

        # 4. Build projection closures (one per BFS layer)
        projections = [
            partial(
                multiple_projection_eff,
                partition=part,
                children_of=children_of,
                pruned_children=pruned_children,
            )
            for part in partitions
        ]
        if self.change_projection_order:
            projections = projections[::-1]

        # 5. Optional momentum / Dykstra states
        if self.uses_p:
            p = jax.tree.map(lambda x: x * 0, inputs)
        if self.uses_q:
            q = jax.tree.map(lambda x: x * 0, inputs)

        # 6. Convergence metric
        def loss_fn(old, new):
            diffs = [
                jnp.mean(jnp.abs(x - y) ** 2)
                for x, y in zip(jax.tree.leaves(old), jax.tree.leaves(new))
            ]
            return sum(diffs) / len(diffs)

        # 7. Scan loop
        if self.uses_p and self.uses_q:
            def step(carry, _):
                vars_, p_, q_ = carry
                new_vars, new_p, new_q = self._step(vars_, *projections, p_, q_)
                loss = loss_fn(vars_, new_vars)
                return (new_vars, new_p, new_q), loss
            (inputs, p, q), losses = jax.lax.scan(
                step, (inputs, p, q), None, length=steps_per_update
            )
        elif self.uses_p:
            def step(carry, _):
                vars_, p_ = carry
                new_vars, new_p = self._step(vars_, *projections, p_)
                loss = loss_fn(vars_, new_vars)
                return (new_vars, new_p), loss
            (inputs, p), losses = jax.lax.scan(
                step, (inputs, p), None, length=steps_per_update
            )
        else:
            def step(carry, _):
                vars_ = carry
                new_vars = self._step(vars_, *projections)
                loss = loss_fn(vars_, new_vars)
                return new_vars, loss
            inputs, losses = jax.lax.scan(
                step, inputs, None, length=steps_per_update
            )

        if self.add_projection_at_end:
            inputs = projections[0](inputs)

        # 8. Extract updated parameters
        new_params = {}
        for name, comp in params.items():
            if comp in inputs:
                new_params[name] = inputs[comp][0]
            else:
                warnings.warn(f"Unused parameter {name}.")
                new_params[name] = params[name].value
        return freeze(new_params), losses.mean()

    @abstractmethod
    def _step(self, vars, *projections):
        """One optimisation step.  Receives the layer projections as
        positional args (same interface as ``optim.Optimizer._step``)."""
        ...


# ────────────────────────────────────────────────────────
#  Concrete optimizers
# ────────────────────────────────────────────────────────

class BackpropProjections(EfficientOptimizer):
    """Cyclic alternating projections, BFS-layer order, no networkx."""

    def _step(self, vars, *projections):
        for proj in projections:
            vars = proj(vars)
        return vars


class BackpropProjectionsMomentum(EfficientOptimizer):
    """Cyclic projections with Nesterov-style momentum."""

    def __init__(self, steps_per_update=50, change_projection_order=False):
        super().__init__(steps_per_update, change_projection_order)
        self.uses_p = True

    def _step(self, vars, *args):
        # last arg is p (momentum buffer)
        *projections, p = args
        vars_look = jax.tree.map(lambda x, d: x + 0.9 * d, vars, p)
        new_vars = vars_look
        for proj in projections:
            new_vars = proj(new_vars)
        new_p = jax.tree.map(lambda x, y: x - y, new_vars, vars)
        return new_vars, new_p


class BackpropDouglasRachford(EfficientOptimizer):
    """Douglas-Rachford splitting over BFS layers.

    Uses relaxation formula  x ← (1-λ)x + λ R(x)  where
    R(x) = 2·P(x) − x.

    Args:
        relaxation: λ parameter (default 0.5).
    """

    def __init__(self, steps_per_update=50, change_projection_order=False, relaxation=0.5):
        super().__init__(steps_per_update, change_projection_order)
        self.relaxation = relaxation

    def _step(self, vars, *projections):
        def reflection(proj, v):
            return jax.tree.map(lambda x, y: 2.0 * x - y, proj(v), v)

        reflected = vars
        for proj in projections:
            reflected = reflection(proj, reflected)

        return jax.tree.map(
            lambda x, r: (1.0 - self.relaxation) * x + self.relaxation * r,
            vars, reflected,
        )


class BackpropDouglasRachfordMomentum(EfficientOptimizer):
    """Douglas-Rachford with Nesterov momentum over BFS layers.

    Args:
        relaxation: λ parameter.
        beta: momentum coefficient.
    """

    def __init__(self, steps_per_update=50, change_projection_order=False,
                 relaxation=0.5, beta=0.9):
        super().__init__(steps_per_update, change_projection_order)
        self.relaxation = relaxation
        self.beta = beta
        self.uses_p = True

    def _step(self, vars, *args):
        *projections, p = args

        def reflection(proj, v):
            return jax.tree.map(lambda x, y: 2.0 * x - y, proj(v), v)

        vars_look = jax.tree.map(lambda x, d: x + self.beta * d, vars, p)

        reflected = vars_look
        for proj in projections:
            reflected = reflection(proj, reflected)

        new_vars = jax.tree.map(
            lambda x, r: (1.0 - self.relaxation) * x + self.relaxation * r,
            vars_look, reflected,
        )
        new_p = jax.tree.map(lambda x, y: x - y, new_vars, vars)
        return new_vars, new_p


class BackpropDykstra(EfficientOptimizer):
    """Dykstra's algorithm over BFS layers."""

    def __init__(self, steps_per_update=50, change_projection_order=False):
        super().__init__(steps_per_update, change_projection_order)
        self.uses_p = True
        self.uses_q = True

    def _step(self, vars, *args):
        # unpack: projections..., p, q
        *projections, p, q = args

        # first half-sweep with p correction
        y = jax.tree.map(lambda x, pp: x + pp, vars, p)
        for proj in projections:
            y = proj(y)
        p_new = jax.tree.map(lambda x, pp, yn: x + pp - yn, vars, p, y)

        # second half-sweep with q correction
        z = jax.tree.map(lambda yn, qq: yn + qq, y, q)
        for proj in projections:
            z = proj(z)
        q_new = jax.tree.map(lambda yn, qq, zn: yn + qq - zn, y, q, z)

        return z, p_new, q_new
