"""
Memory profiling script for conv.py (CNN model)
Profiles the optimizer.update() call with a CNN model to find the largest objects.
"""
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"  # force CPU for profiling clarity

import sys
import jax
import jax.numpy as jnp
import pjax
from pjax import nn, optim
from pjax.core.frozen_dict import FrozenDict, freeze
from pjax.core.computation import Array, Computation, Operation, Parameter, ShapeTransform
import networkx as nx
from functools import partial

# ─── Helpers ──────────────────────────────────────────────────────────────────

def sizeof_fmt(num_bytes):
    """Human-readable byte sizes."""
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:7.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TiB"


def jax_array_report(pytree, label=""):
    """Report all JAX arrays inside a pytree, sorted by size descending."""
    leaves = jax.tree.leaves(pytree)
    arrays = [(l, l.shape, l.dtype, l.nbytes) for l in leaves if hasattr(l, "nbytes")]
    arrays.sort(key=lambda x: -x[3])
    total = sum(a[3] for a in arrays)
    print(f"\n{'='*70}")
    print(f"JAX Array Report: {label}")
    print(f"  Total leaves : {len(leaves)}")
    print(f"  Total arrays : {len(arrays)}")
    print(f"  Total memory : {sizeof_fmt(total)}")
    print(f"{'─'*70}")
    for i, (arr, shape, dtype, nbytes) in enumerate(arrays[:30]):
        print(f"  [{i:3d}] {sizeof_fmt(nbytes):>12s}  shape={str(shape):<25s} dtype={dtype}")
    if len(arrays) > 30:
        print(f"  ... and {len(arrays)-30} more arrays")
    print(f"{'='*70}\n")
    return total

# ─── Model setup (CNN from conv.py) ──────────────────────────────────────────

class CNN_pjax(nn.Module):
    """PJAX CNN model with optional skip connections and max pooling."""
    def __init__(self, hidden_features, in_features, size_2d, classes, skip=True, max_pool=True, stride=1):
        super().__init__()
        self.hidden_features = hidden_features
        self.skip = skip
        self.max_pool = max_pool
        last_f = in_features
        for i, f in enumerate(hidden_features):
            setattr(self, f"conv_{i}", nn.Conv2D(last_f, f, kernel_shape=(3, 3), strides=(1, 1), padding="SAME"))
            setattr(self, f"relu_{i}", nn.ReLU(f))
            last_f = f

        # calculate output features for the final dense layer
        out_features = 0
        current_h, current_w = size_2d, size_2d
        for f in hidden_features:
            current_h = (current_h + stride - 1) // stride
            current_w = (current_w + stride - 1) // stride
            out_features += f if self.max_pool else current_h * current_w * f
        if not self.skip:
            out_features = hidden_features[-1] if max_pool else current_h * current_w * hidden_features[-1]

        self.out = nn.Linear(out_features, classes)

    def __call__(self, x):
        xs = []
        for i in range(len(self.hidden_features)):
            x = getattr(self, f"conv_{i}")(x)
            x = getattr(self, f"relu_{i}")(x)
            if self.max_pool:
                xs.append(pjax.max(x, axis=(1, 2)))
            else:
                xs.append(pjax.reshape(x, (x.shape[0], -1)))

        if self.skip:
            x = pjax.concatenate(xs, axis=-1)
        else:
            x = xs[-1]

        return self.out(x)


key = jax.random.key(0)
# Match parameters from conv.py: [32], in=1, size=28, classes=10
model = CNN_pjax([32], in_features=1, size_2d=28, classes=10, max_pool=True, stride=1)
params = model.init(key)

# Create a small dummy batch: (batch, height, width, channels)
batch_size = 16
x = jax.random.normal(key, (batch_size, 28, 28, 1))
y = jax.random.randint(key, (batch_size,), 0, 10)

print("=" * 70)
print("MODEL PARAMETERS (CNN)")
print("=" * 70)
for name, val in params.items():
    print(f"  {name:<30s} shape={str(val.shape):<20s} dtype={val.dtype}  {sizeof_fmt(val.nbytes)}")
param_total = sum(v.nbytes for v in jax.tree.leaves(params))
print(f"  Total param memory: {sizeof_fmt(param_total)}")

# ─── Profile the optimizer update ─────────────────────────────────────────────

# Use few steps for profiling to keep it fast
optimizer = optim.AlternatingProjections(steps_per_update=2)

def apply_fn(params):
    logits = model.apply(params, x)
    y_one_hot = jax.nn.one_hot(y, num_classes=10)
    # y_one_hot = y_one_hot.astype(jax.numpy.complex64) # As in conv.py?
    # Actually conv.py casts it. Let's do the same.
    y_one_hot = y_one_hot.astype(jax.numpy.complex64)
    return pjax.cross_entropy(logits, y_one_hot)


# ─── Instrument Optimizer.update step-by-step ─────────────────────────────────
print("\n\n" + "=" * 70)
print("PROFILING Optimizer.update() — step by step (CNN)")
print("=" * 70)

# Step 1: Build params & computation
params_inner = {name: Parameter(value, name=name) for name, value in params.items()}
computation = apply_fn(params_inner)
graph = optim.get_graph(computation)

print(f"\n[1] Computation Graph")
print(f"    Nodes: {graph.number_of_nodes()}")
print(f"    Edges: {graph.number_of_edges()}")
print(f"    Node types:")
node_types = {}
for node in graph.nodes:
    t = type(node).__name__
    node_types[t] = node_types.get(t, 0) + 1
for t, count in sorted(node_types.items()):
    print(f"      {t}: {count}")

# Step 2: Build inputs dict
inputs = {}
for node in graph.nodes:
    if isinstance(node, Parameter):
        inputs[node] = [node.value.astype(jnp.complex64)]
    if isinstance(node, Operation):
        inputs[node] = [parent.value.astype(jnp.complex64) for parent in node.parents]
inputs_frozen = freeze(inputs)

print(f"\n[2] Inputs FrozenDict")
print(f"    Keys: {len(inputs_frozen)}")
jax_array_report(inputs_frozen, "inputs (FrozenDict)")

# Step 3: Graph pruning & partitioning
pruned_graph = optim.prune_shape_transforms(graph)
print(f"\n[3] Pruned Graph")
print(f"    Nodes: {pruned_graph.number_of_nodes()}")
print(f"    Edges: {pruned_graph.number_of_edges()}")

try:
    partition_a, partition_b = nx.bipartite.sets(pruned_graph)
    # make sure output node is in partition a
    if any(pruned_graph.out_degree(node) == 0 for node in partition_b):
        partition_a, partition_b = partition_b, partition_a
    print(f"    Partition A: {len(partition_a)} nodes")
    print(f"    Partition B: {len(partition_b)} nodes")
except nx.NetworkXError as e:
    print(f"    Bipartite check failed: {e}")
    partition_a, partition_b = set(), set()


# Step 4: Projections
if partition_a and partition_b:
    projections = [
        partial(optim.multiple_projection, partition=partition_a, graph=graph),
        partial(optim.multiple_projection, partition=partition_b, graph=graph),
    ]

    print(f"\n[4] Running one projection step and measuring result...")
    # NOTE: This might be slow if the graph is huge or operations are expensive (FFTs)
    step_inputs = inputs_frozen
    after_proj_a = projections[0](step_inputs)
    jax_array_report(after_proj_a, "after projection A")

    after_proj_b = projections[1](after_proj_a)
    jax_array_report(after_proj_b, "after projection B (one full step)")

    # Step 5: Check velocity (for momentum-based optimizers)
    print(f"\n[5] Velocity (momentum) structure")
    velocity = jax.tree.map(lambda x: x * 0, inputs_frozen)
    jax_array_report(velocity, "velocity (zero-initialized)")

    # Step 6: loss_fn intermediates
    def loss_fn(old, new):
        diffs = [jnp.mean((x - y) ** 2) for x, y in zip(jax.tree.leaves(old), jax.tree.leaves(new))]
        return sum(diffs) / len(diffs)

    loss = loss_fn(step_inputs, after_proj_b)
    print(f"\n[6] Loss value: {loss}")

    # ─── Summary ─────────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("MEMORY SUMMARY (CNN)")
    print("=" * 70)

    total_params = sum(v.nbytes for v in jax.tree.leaves(params))
    total_inputs = sum(l.nbytes for l in jax.tree.leaves(inputs_frozen) if hasattr(l, "nbytes"))
    total_velocity = sum(l.nbytes for l in jax.tree.leaves(velocity) if hasattr(l, "nbytes"))
    total_after_a = sum(l.nbytes for l in jax.tree.leaves(after_proj_a) if hasattr(l, "nbytes"))
    total_after_b = sum(l.nbytes for l in jax.tree.leaves(after_proj_b) if hasattr(l, "nbytes"))

    items = [
        ("Model params (real)", total_params),
        ("inputs FrozenDict (complex64)", total_inputs),
        ("After projection A", total_after_a),
        ("After projection B", total_after_b),
        ("Velocity (momentum)", total_velocity),
    ]
    items.sort(key=lambda x: -x[1])

    print(f"\n  {'Object':<40s} {'Size':>12s}")
    print(f"  {'─'*40} {'─'*12}")
    for name, size in items:
        print(f"  {name:<40s} {sizeof_fmt(size):>12s}")

    total_live = sum(s for _, s in items)
    print(f"\n  Total tracked live memory: {sizeof_fmt(total_live)}")
    
    # During jax.lax.scan, ALL of these are carried simultaneously:
    scan_carry = total_inputs  # vars (inputs)
    if True:  # momentum variant
        scan_carry += total_velocity
    # Inside one scan step, projections create temporaries
    scan_step_peak = scan_carry + total_after_a + total_after_b  # two projection results
    print(f"\n  Estimated scan carry size:         {sizeof_fmt(scan_carry)}")
    print(f"  Estimated scan step peak memory:   {sizeof_fmt(scan_step_peak)}")
    print(f"  × {optimizer.steps_per_update} steps (scan unroll): {sizeof_fmt(scan_step_peak * optimizer.steps_per_update)}")
    print()
else:
    print("Skipping projection steps due to partitioning failure.")
