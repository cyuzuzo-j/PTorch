"""
Memory benchmark for Torch, Ptorch, and Pjax MLPs.

Measures peak memory of a training step (forward + backward + update) using
tracemalloc.  Each framework configuration runs in a spawned subprocess so
that memory measurements are isolated.

Goal: show that pjax scales O(MNK) while torch/ptorch scale O(MN), where
  M = hidden width (output neurons)
  N = batch size
  K = input features
"""

import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "default"
os.environ["JAX_EXCLUDE_PJRT_PLUGINS"] = "cuda12"

import gc
import importlib
import multiprocessing
import sys
import tracemalloc

import numpy as np
import pandas as pd
import torch
import torch.nn as tnn
import torch.nn.functional as F
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../frameworks")))

from ptorch.nn.modules import Linear as PLinear, SumReLU
from ptorch.core.ops import CrossEntropyProjection

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class TorchMLP(tnn.Module):
    """Standard PyTorch MLP with ReLU activations."""

    def __init__(self, hidden_dim, depth, in_features, classes):
        super().__init__()
        self.hidden_layers = tnn.ModuleList()
        last = in_features
        for _ in range(depth):
            self.hidden_layers.append(tnn.Linear(last, hidden_dim))
            self.hidden_layers.append(tnn.ReLU())
            last = hidden_dim
        self.out = tnn.Linear(last, classes)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        for i in range(0, len(self.hidden_layers), 2):
            x = self.hidden_layers[i + 1](self.hidden_layers[i](x))
        return self.out(x)


class PtorchMLP(tnn.Module):
    """Ptorch MLP using projection-based Linear layers."""

    def __init__(self, hidden_dim, depth, in_features, classes):
        super().__init__()
        self.hidden_layers = tnn.ModuleList()
        last = in_features
        for _ in range(depth):
            self.hidden_layers.append(PLinear(last, hidden_dim, residual=False))
            self.hidden_layers.append(SumReLU())
            last = hidden_dim
        self.out = PLinear(last, classes)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        for i in range(0, len(self.hidden_layers), 2):
            x = self.hidden_layers[i + 1](self.hidden_layers[i](x))
        return self.out(x)


def load_pjax():
    """Dynamically load pjax_orr and return (JAXMLP, nn, api) or None."""
    try:
        pjax_mod = importlib.import_module("pjax_orr")
        nn = importlib.import_module("pjax_orr.nn")
        api = importlib.import_module("pjax_orr.core.api")
    except ImportError:
        return None

    class JAXMLP(nn.Module):
        def __init__(self, hidden_dim, depth, in_features, classes):
            super().__init__()
            last = in_features
            for i in range(depth):
                setattr(self, f"linear_{i}", nn.Linear(last, hidden_dim))
                setattr(self, f"relu_{i}", nn.ReLU(hidden_dim))
                last = hidden_dim
            self.depth = depth
            self.out = nn.Linear(last, classes)

        def get_params(self, key):
            try:
                return self.init(key)
            except Exception:
                return {}

        def __call__(self, x):
            x = x.reshape((x.shape[0], -1))
            for i in range(self.depth):
                x = getattr(self, f"relu_{i}")(getattr(self, f"linear_{i}")(x))
            return self.out(x)

    return JAXMLP, nn, api


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

WARMUP_STEPS = 3
MEASURE_STEPS = 10


def _traced_peak_mb(fn):
    """Run *fn*, return peak tracemalloc memory in MB."""
    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()
    fn()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak_bytes / (1024 * 1024)


def _dummy_iter(N, K):
    """Infinite iterator yielding random (x, y) batches."""
    while True:
        yield (
            np.random.randn(N, K).astype(np.float32),
            np.random.randint(0, 10, (N,)),
        )


# ---------------------------------------------------------------------------
# Per-framework measurement functions
# ---------------------------------------------------------------------------

def measure_torch(width, depth, train_iter, batch_size=128, in_features=784, classes=10):
    device = torch.device("cpu")

    model = TorchMLP(width, depth, in_features, classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss_fn = tnn.CrossEntropyLoss()

    def _step():
        x_np, y_np = next(train_iter)
        x = torch.tensor(x_np, dtype=torch.float32, device=device)
        y = torch.tensor(y_np, dtype=torch.long, device=device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

    # Warmup (outside measurement window)
    for _ in range(WARMUP_STEPS):
        _step()

    def _measure():
        for _ in range(MEASURE_STEPS):
            _step()

    peak_mb = _traced_peak_mb(_measure)

    del model, optimizer
    gc.collect()
    return peak_mb


def measure_ptorch(width, depth, train_iter, batch_size=128, in_features=784, classes=10):
    device = torch.device("cpu")

    model = PtorchMLP(width, depth, in_features, classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    def _step():
        x_np, y_np = next(train_iter)
        x = torch.tensor(x_np, dtype=torch.float32, device=device)
        y = torch.tensor(y_np, dtype=torch.long, device=device)
        y_oh = F.one_hot(y, num_classes=classes).float()
        optimizer.zero_grad()
        projected = CrossEntropyProjection.apply(model(x), y_oh)
        projected.sum().backward()
        optimizer.step()

    # Warmup (outside measurement window)
    for _ in range(WARMUP_STEPS):
        _step()

    def _measure():
        for _ in range(MEASURE_STEPS):
            _step()

    peak_mb = _traced_peak_mb(_measure)

    del model, optimizer
    gc.collect()
    return peak_mb


def measure_pjax(width, depth, train_iter, batch_size=128, in_features=784, classes=10):
    pjax_deps = load_pjax()
    if pjax_deps is None:
        return 0.0
    JAXMLP, nn, api = pjax_deps

    model = JAXMLP(width, depth, in_features, classes)
    key = jax.random.PRNGKey(0)
    params = model.get_params(key)
    opt_state = {}

    @jax.jit
    def step_fn(p, o_state, x, y):
        def apply_fn(pp):
            pred = model.apply(pp, x)
            y_oh = jax.nn.one_hot(y, classes)
            return jnp.mean(api.cross_entropy(pred, y_oh))

        loss, grads = jax.value_and_grad(apply_fn)(p)
        new_p = jax.tree_util.tree_map(lambda param, g: param - 0.1 * g, p, grads)
        return new_p, o_state, loss

    # Warmup / JIT compile (outside measurement window)
    for _ in range(WARMUP_STEPS):
        x_np, y_np = next(train_iter)
        params, opt_state, _ = step_fn(params, opt_state, jnp.array(x_np), jnp.array(y_np))
    jax.block_until_ready(params)

    def _measure():
        nonlocal params, opt_state
        for _ in range(MEASURE_STEPS):
            x_np, y_np = next(train_iter)
            params, opt_state, _ = step_fn(params, opt_state, jnp.array(x_np), jnp.array(y_np))
        jax.block_until_ready(params)

    return _traced_peak_mb(_measure)


# ---------------------------------------------------------------------------
# Subprocess isolation
# ---------------------------------------------------------------------------

MEASURE_FNS = {
    "torch": measure_torch,
    "ptorch": measure_ptorch,
    "pjax": measure_pjax,
}


def _run_in_subprocess(func_name, M, D, N, K, return_dict):
    """Entry point for spawned subprocess."""
    train_iter = _dummy_iter(N, K)
    return_dict[func_name] = MEASURE_FNS[func_name](
        M, D, train_iter, batch_size=N, in_features=K
    )


def run_isolated(func_name, M, D, N, K):
    """Run a single measurement in an isolated subprocess and return peak MB."""
    ctx = multiprocessing.get_context("spawn")
    manager = ctx.Manager()
    return_dict = manager.dict()
    p = ctx.Process(target=_run_in_subprocess, args=(func_name, M, D, N, K, return_dict))
    p.start()
    p.join()
    if p.exitcode != 0:
        return 0.0
    return return_dict.get(func_name, 0.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Depth = 1 to isolate a single K -> M layer's memory behaviour
    D = 1
    M_vals = [500, 2500, 5000]
    N_vals = [8, 64, 256]
    K_vals = [500, 2500, 5000]

    header = f"{'M (Out)':<8} | {'N (Batch)':<9} | {'K (In)':<8} | {'torch (MB)':<12} | {'ptorch (MB)':<12} | {'pjax (MB)':<12}"
    print("Starting Memory Benchmark (MB)")
    print(header)
    print("-" * len(header))

    results = []
    for M in M_vals:
        for N in N_vals:
            for K in K_vals:
                mem = {fw: run_isolated(fw, M, D, N, K) for fw in MEASURE_FNS}

                print(f"{M:<8} | {N:<9} | {K:<8} | {mem['torch']:<12.2f} | {mem['ptorch']:<12.2f} | {mem['pjax']:<12.2f}")
                results.append({"M": M, "N": N, "K": K, **{f"{fw}_mb": mem[fw] for fw in MEASURE_FNS}})

    df = pd.DataFrame(results)
    df.to_csv("memory_benchmark_results.csv", index=False)
    print(f"\nResults saved to memory_benchmark_results.csv")


if __name__ == "__main__":
    main()
