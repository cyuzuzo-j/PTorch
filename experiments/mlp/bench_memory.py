"""
bench_memory.py
================
Peak-memory benchmark for a single linear-layer projection.

Reproduces the curve in ``images/mlp/memory_vs_k_report.png``: peak heap
memory of a training step (forward + backward + optimizer.step) as the
input dimension ``N_in`` is swept, with the output dimension ``M`` and the
batch size ``B`` held fixed.

Expected scaling::

    pjax    : O(M * B * K)            (3D tensor materialised in projection)
    ptorch  : O(M*K + M*B + K*B)      (only 2D operands)
    torch   : O(M*K)                  (just the weight matrix)

So pjax grows as ~K^2 (with M, B fixed), while ptorch tracks torch's
O(M*K) asymptote.

Each measurement runs in a fresh ``spawn`` subprocess, so allocations from
one framework don't bleed into the next, and the driver process stays at a
near-zero baseline. Memory is measured on CPU two ways:

* ``tracemalloc`` — captures Python-side heap. Reproduces the methodology of
  the original benchmark in commit de32989. NaN-safe but only sees
  Python-allocator memory; the torch CPU caching allocator goes through
  ``operator new`` and so is *invisible* to tracemalloc.
* ``ru_maxrss`` — peak resident set size of the whole subprocess. Captures
  every allocation including torch's C++ tensors. This is the column that
  matches the figure in the paper. Reported as ``peak - baseline`` where
  ``baseline`` is the RSS right after the framework's imports.

``pjax_orr`` (and JAX) are optional; if either is missing the pjax column
is reported as NaN.

Usage::

    python experiments/mlp/bench_memory.py
    python experiments/mlp/bench_memory.py --K 500 1000 2000 --frameworks torch ptorch
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "default")

import argparse
import gc
import multiprocessing
import os.path as osp
import resource
import sys
import tracemalloc

import numpy as np
import pandas as pd

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "../..")))
sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "../../frameworks")))

WARMUP_STEPS = 3
MEASURE_STEPS = 5


# ── helpers ─────────────────────────────────────────────────────────────────

def _ru_maxrss_mb():
    """Peak RSS of this process so far, in MB.

    ``ru_maxrss`` is reported in kilobytes on Linux and in bytes on macOS;
    we normalise to MB.
    """
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":  # macOS reports bytes
        return rss_kb / (1024 * 1024)
    return rss_kb / 1024


def _run_with_tracking(warmup_fn, measure_fn):
    """Run warmup + measurement and return (tracemalloc peak MB, RSS delta MB).

    The RSS baseline is captured *before* warmup, so the delta reflects all
    allocations triggered by the model + warmup + measurement steps. This
    matters because ``ru_maxrss`` is a lifetime high-water mark — measuring
    after warmup already covers the peak and yields zero delta.
    """
    gc.collect()
    rss_before = _ru_maxrss_mb()
    warmup_fn()
    tracemalloc.start()
    tracemalloc.reset_peak()
    measure_fn()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = _ru_maxrss_mb()
    return peak_bytes / (1024 * 1024), max(rss_after - rss_before, 0.0)


def _data(B, K, classes, seed=0):
    rng = np.random.default_rng(seed)
    while True:
        yield (
            rng.standard_normal((B, K), dtype=np.float32),
            rng.integers(0, classes, (B,), dtype=np.int64),
        )


# ── per-framework measurement (each runs in its own subprocess) ─────────────

def measure_torch(M, B, K):
    import torch
    import torch.nn as tnn

    layer = tnn.Linear(K, M)
    optim = torch.optim.SGD(layer.parameters(), lr=0.01)
    loss_fn = tnn.CrossEntropyLoss()
    it = _data(B, K, classes=M)

    def step():
        x_np, y_np = next(it)
        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)
        optim.zero_grad()
        loss_fn(layer(x), y).backward()
        optim.step()

    return _run_with_tracking(
        warmup_fn=lambda: [step() for _ in range(WARMUP_STEPS)],
        measure_fn=lambda: [step() for _ in range(MEASURE_STEPS)],
    )


def measure_ptorch(M, B, K):
    import torch
    import torch.nn.functional as F
    from ptorch.nn.modules import Linear as PLinear
    from ptorch.core.ops import CrossEntropyProjection
    from ptorch import config as ptorch_config

    ptorch_config.config.use_projections = True

    layer = PLinear(K, M)
    optim = torch.optim.SGD(layer.parameters(), lr=0.01)
    it = _data(B, K, classes=M)

    def step():
        x_np, y_np = next(it)
        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)
        y_oh = F.one_hot(y, num_classes=M).float()
        optim.zero_grad()
        proj = CrossEntropyProjection.apply(layer(x), y_oh)
        proj.sum().backward()
        optim.step()

    return _run_with_tracking(
        warmup_fn=lambda: [step() for _ in range(WARMUP_STEPS)],
        measure_fn=lambda: [step() for _ in range(MEASURE_STEPS)],
    )


def measure_pjax(M, B, K):
    """Measure peak memory of one pjax projection-based update.

    pjax's projection operator for ``dot`` (``bilinear_proj``) is vmapped twice
    inside ``matmul``, so each scalar bilinear projection runs over a (B, M)
    grid of K-vectors — that's where the O(B*M*K) materialisation comes from.
    To exercise this path we have to drive the model through
    ``Optimizer.update``; using plain ``jax.value_and_grad`` would compile to
    a regular matmul with no 3D blowup.
    """
    try:
        import jax
        import jax.numpy as jnp
        from pjax_orr import nn as pnn
        from pjax_orr.core import api as papi
        from pjax_orr.optim import AlternatingProjections
    except ImportError:
        return float("nan"), float("nan")

    class JAXLinear(pnn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.linear = pnn.Linear(in_features, out_features)

        def __call__(self, x):
            return self.linear(x)

    model = JAXLinear(K, M)
    key = jax.random.PRNGKey(0)
    params = model.init(key)
    optimizer = AlternatingProjections(steps_per_update=1)
    it = _data(B, K, classes=M)

    def step():
        nonlocal params
        x_np, y_np = next(it)
        x = jnp.asarray(x_np)
        y_oh = jax.nn.one_hot(jnp.asarray(y_np), M)

        def fun(p):
            logits = model.apply(p, x)
            return papi.cross_entropy(logits, y_oh)

        params, _ = optimizer.update(fun, params)
        jax.block_until_ready(params)

    return _run_with_tracking(
        warmup_fn=lambda: [step() for _ in range(WARMUP_STEPS)],
        measure_fn=lambda: [step() for _ in range(MEASURE_STEPS)],
    )


MEASURE_FNS = {
    "torch":  measure_torch,
    "ptorch": measure_ptorch,
    "pjax":   measure_pjax,
}


# ── subprocess isolation ────────────────────────────────────────────────────

def _entry(name, M, B, K, ret):
    ret[name] = MEASURE_FNS[name](M, B, K)


def run_isolated(name, M, B, K):
    """Returns (tracemalloc_peak_mb, rss_delta_mb)."""
    ctx = multiprocessing.get_context("spawn")
    mgr = ctx.Manager()
    ret = mgr.dict()
    p = ctx.Process(target=_entry, args=(name, M, B, K, ret))
    p.start()
    p.join()
    if p.exitcode != 0:
        return float("nan"), float("nan")
    return ret.get(name, (float("nan"), float("nan")))


# ── main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--M", type=int, default=500, help="output dim (held fixed)")
    p.add_argument("--B", type=int, default=256, help="batch size (held fixed)")
    p.add_argument("--K", type=int, nargs="+",
                   default=[100, 250, 500, 1000, 2000, 4000, 8000],
                   help="input dimensions to sweep")
    p.add_argument("--frameworks", nargs="+",
                   default=["torch", "ptorch", "pjax"])
    p.add_argument("--out", default=osp.join(osp.dirname(__file__),
                                              "results", "memory_vs_k.csv"))
    args = p.parse_args()

    os.makedirs(osp.dirname(args.out), exist_ok=True)

    print(f"M={args.M}  B={args.B}  K∈{args.K}  frameworks={args.frameworks}")
    print("Reporting peak RSS delta (MB) [tracemalloc peak in brackets]")
    cols = "  ".join(f"{fw:>18}" for fw in args.frameworks)
    print(f"{'K':>6}  {cols}")
    print("-" * (8 + 20 * len(args.frameworks)))

    rows = []
    for K in args.K:
        mem = {fw: run_isolated(fw, args.M, args.B, K) for fw in args.frameworks}
        cells = "  ".join(
            f"{mem[fw][1]:>9.2f} [{mem[fw][0]:>5.2f}]" for fw in args.frameworks
        )
        print(f"{K:>6}  {cells}")
        row = {"M": args.M, "B": args.B, "K": K}
        for fw in args.frameworks:
            tr_mb, rss_mb = mem[fw]
            row[f"{fw}_tracemalloc_mb"] = tr_mb
            row[f"{fw}_rss_mb"] = rss_mb
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
