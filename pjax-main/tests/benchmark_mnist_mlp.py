
import time
import jax
import jax.numpy as jnp
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import pjax
from pjax import nn, optim
from pjax.core import api
from pjax.core import ops
from pjax.core.computation import vmap, array
from pjax.core.api import broadcast_to, expand_dims, squeeze
from pjax.optim_efficient import get_graph
from functools import partial

# Define synthetic data generator to avoid torch/torchvision overhead
def synthetic_mnist_iterator(batch_size=128):
    key = jax.random.key(42)
    while True:
        key, k1, k2 = jax.random.split(key, 3)
        x = jax.random.uniform(k1, (batch_size, 28, 28, 1))
        y = jax.random.randint(k2, (batch_size,), 0, 10)
        yield x, y

# --- Model Definition (Copied from comparison.py) ---
class MLP_pjax(nn.Module):
    def __init__(self, hidden_features, in_features, classes, skip=True, inefff=True):
        super().__init__()
        self.hidden_features = hidden_features
        self.skip = skip
        last_f = in_features
        for i, f in enumerate(hidden_features):
            setattr(self, f"dense_{i}", nn.Linear(last_f, f))
            setattr(self, f"relu_{i}", nn.ReLU(f))
            last_f = f

        out_features = sum(hidden_features) if skip else hidden_features[-1]
        self.out = nn.Linear(out_features, classes) if not inefff else nn.LinearOld(out_features, classes)

    def __call__(self, x):
        # Flatten input for MLP
        x = pjax.reshape(x, (x.shape[0], -1))
        xs = []
        for i in range(len(self.hidden_features)):
            x = getattr(self, f"dense_{i}")(x)
            x = getattr(self, f"relu_{i}")(x)
            xs.append(x)

        if self.skip:
            x = pjax.concatenate(xs, axis=-1)

        return self.out(x)


def log(msg):
    print(msg, flush=True)
    with open("benchmark_results.txt", "a") as f:
        f.write(msg + "\n")

# --- Benchmark Runner ---
def run_benchmark(setup_name, steps=50, profile=False, old=False):
    log(f"\nRunning Benchmark: {setup_name}")
    
    # Setup Data
    BATCH_SIZE = 64
    train_iter = synthetic_mnist_iterator(batch_size=BATCH_SIZE)
    
    # Setup Model
    input_features = 28 * 28
    classes = 10
    model = MLP_pjax(hidden_features=[256, 256], in_features=input_features, classes=classes, skip=True, inefff=old)
    
    # Init Params
    key = jax.random.key(0)
    params = model.init(key)
    
    # Setup Optimizer
    optimizer = optim.AlternatingProjections(steps_per_update=1)
    
    # JIT Compile / First Step
    log("  Compiling / First step...")
    x, y = next(train_iter)
    
    def apply_fn(params, x, y):
        def model_fn(p):
            pred = model.apply(p, x)
            return pjax.cross_entropy(pred, jax.nn.one_hot(y, classes))
        
        # We need to construct the graph to count repeats
        loss_computation = model_fn(params)
        return loss_computation
        
    # Get Graph Statistics
    loss_comp = apply_fn(params, array(jnp.array(x)), jnp.array(y))
    graph = get_graph(loss_comp)
    
    total_nodes = len(graph.nodes)
    repeat_nodes = sum(1 for n in graph.nodes if hasattr(n, 'name') and n.name == 'repeat')
    log(f"  Graph Stats: Total Nodes={total_nodes}, Repeat Nodes={repeat_nodes}")
    
    # Run Training Loop for timing
    def train_step(params, x, y):
        def model_fn(p):
            pred = model.apply(p, x)
            return pjax.cross_entropy(pred, jax.nn.one_hot(y, classes))
        return optimizer.update(model_fn, params)[0]

    # Warmup
    x_jax, y_jax = jnp.array(x), jnp.array(y)
    params = train_step(params, x_jax, y_jax) # JITs here usually if jitted
    
    # Benchmark Loop
    log(f"  Running {steps} steps...")
    start_time = time.time()
    for i in range(steps):
        if i % 10 == 0:
             log(f"    Step {i}/{steps}...")
        x, y = next(train_iter)
        params = train_step(params, jnp.array(x), jnp.array(y))

    # Block to ensure timing is accurate for async dispatch
    jax.block_until_ready(list(params.values())[0])

    end_time = time.time()

    if profile:
        profile_filename = f"memory_{setup_name.replace(' ', '_').replace('(', '').replace(')', '')}.prof"
        jax.profiler.save_device_memory_profile(profile_filename)
        log(f"  Saved memory profile to {profile_filename}")

    avg_time = (end_time - start_time) / steps
    log(f"  Time per step: {avg_time*1000:.2f} ms")
    
    return {
        "setup": setup_name,
        "time_ms": avg_time * 1000,
        "total_nodes": total_nodes,
        "repeat_nodes": repeat_nodes
    }

if __name__ == "__main__":
    # Clear previous results
    with open("benchmark_results.txt", "w") as f:
        f.write("Starting benchmark...\n")

    results = []
    
    # 1. Run with Efficient Matmul (Default)
    # Using fewer steps to be faster
    results.append(run_benchmark("Efficient Matmul (New)", steps=20, profile=True))
    
    # 2. Run with Inefficient Matmul (Patched)
    log("\nPatching pjax.matmul with inefficient implementation...")
    
    results.append(run_benchmark("Inefficient Matmul (Old)", steps=20, profile=True, old=True))
        
    # Print Comparison Table
    log("\n" + "="*60)
    log(f"{'Setup':<25} | {'Time (ms)':<10} | {'Nodes':<8} | {'Repeats':<8}")
    log("-" * 60)
    for r in results:
        log(f"{r['setup']:<25} | {r['time_ms']:<10.2f} | {r['total_nodes']:<8} | {r['repeat_nodes']:<8}")
    log("="*60)
