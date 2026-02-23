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

from abc import ABC, abstractmethod
from functools import partial

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
#  Core helpers
# ────────────────────────────────────────────────────────

def _projection_indexed(idx, inputs_list, node_index, children_of_idx, computation):
    """Project a single node using integer-indexed list instead of Computation-keyed dict."""
    if isinstance(computation, Array):
        return inputs_list
    # Gather outputs from children
    kids_idx = children_of_idx.get(idx, [])
    if not kids_idx:
        output = None
    else:
        vals = []
        for child_idx, child_node, parent_positions in kids_idx:
            child_inputs = inputs_list[child_idx]
            if isinstance(child_node, Operation):
                for pos in parent_positions:
                    vals.append(child_inputs[pos])
            elif isinstance(child_node, ShapeTransform):
                # Re-evaluate ShapeTransform inverse using current values
                transform_inputs = [inputs_list[node_index[p]][0] if isinstance(p, Parameter)
                                    else inputs_list[node_index[p]] for p in child_node.parents]
                # flatten to single values
                flat_ti = []
                for ti in transform_inputs:
                    flat_ti.append(ti[0] if isinstance(ti, list) else ti)
                child_out_vals = []
                for gchild_idx, gchild_node, gparent_positions in children_of_idx.get(child_idx, []):
                    gchild_inputs = inputs_list[gchild_idx]
                    if isinstance(gchild_node, Operation):
                        for pos in gparent_positions:
                            child_out_vals.append(gchild_inputs[pos])
                if not child_out_vals:
                    continue
                child_output = jnp.sum(jnp.stack(child_out_vals), axis=0) / len(child_out_vals)
                inv_vals = child_node.inverse(*flat_ti, child_output)
                if isinstance(inv_vals, jnp.ndarray):
                    inv_vals = [inv_vals]
                for pos in parent_positions:
                    vals.append(inv_vals[pos])
        if not vals:
            output = None
        else:
            stacked = jnp.stack(vals)
            output = jnp.sum(stacked, axis=0) / len(vals)

    if output is None:
        return inputs_list

    projected = computation.projection(*inputs_list[idx], output)
    return inputs_list[:idx] + [list(projected)] + inputs_list[idx + 1:]


def _scatter_params(inputs_list, params_flat, param_indices):
    """Write params_flat values back into the inputs_list Parameter slots."""
    for i, idx in enumerate(param_indices):
        inputs_list = inputs_list[:idx] + [[params_flat[i]]] + inputs_list[idx + 1:]
    return inputs_list


def _gather_params(inputs_list, param_indices):
    """Extract Parameter slot values from inputs_list as a flat list."""
    return [inputs_list[idx][0] for idx in param_indices]


def _project_partition(
    params_flat: list,
    static_slots: list,              # fixed non-Parameter slots (Array/ShapeTransform placeholders)
    param_indices: list,             # [int]  positions of Parameter nodes in topo_order
    partition_indices: list,
    children_of_idx: dict,
    update_indices: list,
    node_index: dict,
):
    """Rebuild inputs_list from static skeleton + current params, run projections,
    return only the updated params_flat.  op_slots are never carried."""
    inputs_list = _scatter_params(static_slots, params_flat, param_indices)

    for idx, node in partition_indices:
        inputs_list = _projection_indexed(idx, inputs_list, node_index, children_of_idx, node)

    # Refresh Operation children so subsequent partitions see updated inputs
    for idx, node in update_indices:
        new_vals = []
        for p in node.parents:
            pidx = node_index[p]
            pval = inputs_list[pidx]
            new_vals.append((pval[0] if isinstance(pval, list) else pval).astype(jnp.bfloat16))
        inputs_list = inputs_list[:idx] + [new_vals] + inputs_list[idx + 1:]

    return _gather_params(inputs_list, param_indices)


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

        # 1. Build computation graph
        param_nodes = {name: Parameter(value, name=name) for name, value in params.items()}
        computation = fun(param_nodes)

        # 2. Build graph topology
        topo_order = _topo_sort(computation)
        children_of = _children_map(topo_order)
        pruned_children = _pruned_children_map(topo_order, children_of)
        partitions = _bfs_partitions(topo_order, children_of)

        # 3. Stable integer index: Computation → int (pure Python, never traced)
        node_index = {node: i for i, node in enumerate(topo_order)}

        # 4. Build full inputs_list (stable pytree structure)
        #    inputs_list[i] = [array, ...]
        #      - Parameter:      [param_value]
        #      - Operation:      [parent1_val, parent2_val, ...]
        #      - Array/Shape:    [node_value]  (placeholder, never projected)
        inputs_list = []
        for node in topo_order:
            if isinstance(node, Parameter):
                inputs_list.append([node.value.astype(jnp.bfloat16)])
            elif isinstance(node, Operation):
                inputs_list.append([p.value.astype(jnp.bfloat16) for p in node.parents])
            else:
                inputs_list.append([node.value.astype(jnp.bfloat16)])

        # 5. Parameter indices (stable, never changes across steps)
        param_indices = [node_index[comp] for comp in param_nodes.values()]

        # 6. Build children_of_idx: int → [(child_idx, child_node, [parent_positions])]
        children_of_idx: dict[int, list] = {i: [] for i in range(len(topo_order))}
        for node in topo_order:
            if isinstance(node, (ShapeTransform, Array)):
                continue
            idx = node_index[node]
            for child in children_of[node]:
                child_idx = node_index[child]
                parent_positions = [i for i, p in enumerate(child.parents) if p == node]
                children_of_idx[idx].append((child_idx, child, parent_positions))

        # 7. Build a *static* skeleton: same structure as inputs_list but
        #    Parameter slots zeroed out (they will be filled from params_flat
        #    at the start of every _project_partition call).  This skeleton
        #    never enters the scan carry, so its shape never needs to be stable.
        static_slots = list(inputs_list)  # shallow copy; Parameter slots will be overwritten

        # 8. Build projection closures.
        #    Each closure signature: params_flat → new_params_flat
        #    The static_slots skeleton and all graph metadata are captured by closure.
        def _make_partition_closures(partitions):
            result = []
            for part in partitions:
                part_indices = sorted(
                    [(node_index[n], n) for n in part if not isinstance(n, (Array, ShapeTransform))],
                    key=lambda x: x[0],
                )
                update_set: dict[int, Computation] = {}
                for idx, node in part_indices:
                    for child in pruned_children.get(node, []):
                        if isinstance(child, Operation):
                            cidx = node_index[child]
                            update_set[cidx] = child
                update_indices = sorted(update_set.items(), key=lambda x: x[0])

                result.append(
                    partial(
                        _project_partition,
                        static_slots=static_slots,
                        param_indices=param_indices,
                        partition_indices=part_indices,
                        children_of_idx=children_of_idx,
                        update_indices=update_indices,
                        node_index=node_index,
                    )
                )
            return result

        projections = _make_partition_closures(partitions)
        if self.change_projection_order:
            projections = projections[::-1]

        # 9. Initial params_flat — the ONLY scan carry (uniform list of arrays)
        params_flat = _gather_params(inputs_list, param_indices)

        # 10. Momentum / Dykstra zero buffers — same structure as params_flat
        zeros_flat = [jnp.zeros_like(p) for p in params_flat]
        if self.uses_p:
            p = list(zeros_flat)
        if self.uses_q:
            q = list(zeros_flat)

        # 11. Convergence metric
        def loss_fn(old_pf, new_pf):
            diffs = [jnp.mean(jnp.abs(x - y) ** 2) for x, y in zip(old_pf, new_pf)]
            return sum(diffs) / len(diffs)

        # 12. Scan loop — carry is ONLY params_flat (+ optional p, q).
        #     op_slots are rebuilt from scratch inside each projection closure;
        #     they never appear in the carry so their shapes can vary freely.
        if self.uses_p and self.uses_q:
            def step(carry, _):
                pf, p_, q_ = carry
                new_pf, new_p, new_q = self._step(pf, projections, p_, q_)
                loss = loss_fn(pf, new_pf)
                return (new_pf, new_p, new_q), loss
            (params_flat, p, q), losses = jax.lax.scan(
                step, (params_flat, p, q), None, length=steps_per_update
            )
        elif self.uses_p:
            def step(carry, _):
                pf, p_ = carry
                new_pf, new_p = self._step(pf, projections, p_)
                loss = loss_fn(pf, new_pf)
                return (new_pf, new_p), loss
            (params_flat, p), losses = jax.lax.scan(
                step, (params_flat, p), None, length=steps_per_update
            )
        else:
            def step(carry, _):
                pf = carry
                new_pf = self._step(pf, projections)
                loss = loss_fn(pf, new_pf)
                return new_pf, loss
            params_flat, losses = jax.lax.scan(
                step, params_flat, None, length=steps_per_update
            )

        if self.add_projection_at_end:
            params_flat = projections[0](params_flat)

        # 13. Rebuild named params dict
        new_params = {}
        for i, name in enumerate(param_nodes):
            new_params[name] = params_flat[i]
        return freeze(new_params), losses.mean()

    @abstractmethod
    def _step(self, params_flat, projections, *aux):
        """One optimisation step.

        Args:
            params_flat: list[array] — one array per Parameter.
            projections: list of callables, each with signature
                         ``params_flat → new_params_flat``.
            *aux:        optional momentum / Dykstra buffers
                         (same list[array] structure as params_flat).
        """
        ...


# ────────────────────────────────────────────────────────
#  Concrete optimizers
# ────────────────────────────────────────────────────────

class BackpropProjections(EfficientOptimizer):
    """Cyclic alternating projections, BFS-layer order, no networkx."""

    def _step(self, params_flat, projections):
        for proj in projections:
            params_flat = proj(params_flat)
        return params_flat


class BackpropProjectionsMomentum(EfficientOptimizer):
    """Cyclic projections with Nesterov-style momentum."""

    def __init__(self, steps_per_update=50, change_projection_order=False):
        super().__init__(steps_per_update, change_projection_order)
        self.uses_p = True

    def _step(self, params_flat, projections, p):
        params_look = [x + 0.9 * d for x, d in zip(params_flat, p)]
        new_pf = params_look
        for proj in projections:
            new_pf = proj(new_pf)
        new_p = [x - y for x, y in zip(new_pf, params_flat)]
        return new_pf, new_p


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

    def _step(self, params_flat, projections):
        def reflection(proj, pf):
            new_pf = proj(pf)
            return [2.0 * px - x for px, x in zip(new_pf, pf)]

        reflected = params_flat
        for proj in projections:
            reflected = reflection(proj, reflected)

        lam = self.relaxation
        return [(1.0 - lam) * x + lam * r for x, r in zip(params_flat, reflected)]


class BackpropDouglasRachfordMomentum(EfficientOptimizer):
    """Douglas-Rachford with Nesterov momentum over BFS layers."""

    def __init__(self, steps_per_update=50, change_projection_order=False,
                 relaxation=0.5, beta=0.9):
        super().__init__(steps_per_update, change_projection_order)
        self.relaxation = relaxation
        self.beta = beta
        self.uses_p = True

    def _step(self, params_flat, projections, p):
        def reflection(proj, pf):
            new_pf = proj(pf)
            return [2.0 * px - x for px, x in zip(new_pf, pf)]

        params_look = [x + self.beta * d for x, d in zip(params_flat, p)]
        reflected = params_look
        for proj in projections:
            reflected = reflection(proj, reflected)

        lam = self.relaxation
        new_pf = [(1.0 - lam) * x + lam * r for x, r in zip(params_look, reflected)]
        new_p = [x - y for x, y in zip(new_pf, params_flat)]
        return new_pf, new_p


class BackpropDykstra(EfficientOptimizer):
    """Dykstra's algorithm over BFS layers."""

    def __init__(self, steps_per_update=50, change_projection_order=False):
        super().__init__(steps_per_update, change_projection_order)
        self.uses_p = True
        self.uses_q = True

    def _step(self, params_flat, projections, p, q):
        # first half-sweep with p correction
        y = [x + pp for x, pp in zip(params_flat, p)]
        for proj in projections:
            y = proj(y)
        p_new = [x + pp - yn for x, pp, yn in zip(params_flat, p, y)]

        # second half-sweep with q correction
        z = [yn + qq for yn, qq in zip(y, q)]
        for proj in projections:
            z = proj(z)
        q_new = [yn + qq - zn for yn, qq, zn in zip(y, q, z)]

        return z, p_new, q_new
