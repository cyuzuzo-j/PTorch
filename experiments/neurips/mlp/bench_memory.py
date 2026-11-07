"""
bench_memory.py
================
Peak-memory benchmark for a single linear-layer projection.

Reproduces the curve in ``images/memory_vs_k.pdf`` and ``images/memory_vs_batch.pdf``: peak memory of a
training step (forward + backward + optimizer.step) as the input dimension
``K`` is swept, with the output dimension ``M`` and the batch size ``B``
held fixed. Each framework trains with the optimizer and loss of the YAML's
``torch:`` / ``ptorch:`` / ``pjax:`` blocks (the ones the training benches
use), so optimizer state such as Adam's moments is counted.

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

* ``device`` (``device: cuda``) — peak bytes held by the framework's GPU
  allocator over model init and the measured steps (``max_memory_allocated``;
  JAX ``peak_bytes_in_use``). XLA's peak cannot be reset, so JAX's also spans
  the warm-up; torch's skips it, as that is where Inductor compiles and
  autotunes (its benchmarker allocates an L2-sized flush buffer). The warm-up
  has the measured shapes, so only compile memory differs. This is the column
  for the paper figure: the CPU RSS delta also counts torch.compile/inductor
  and JAX compiler memory, which inflates ptorch at small K.

``pjax`` (third_party/pjax submodule) and JAX are optional; if either is missing the pjax column
is reported as NaN.

Usage::

    python -m experiments.neurips.mlp.bench_memory --config experiments/neurips/mlp/memory_config.yaml
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "default")

import argparse
import gc
import multiprocessing
import resource
import sys
import tracemalloc

import numpy as np
import pandas as pd

from experiments.neurips import common

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


def _torch_device_peak_mb(device):
    """torch CUDA allocator peak (MB) since the previous call; starts a new window."""
    if device != "cuda":
        return float("nan")
    import torch
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    torch.cuda.reset_peak_memory_stats()
    return peak


def _run_with_tracking(step, device_peak_fn):
    """Run WARMUP_STEPS + MEASURE_STEPS of ``step`` and return
    (tracemalloc peak MB, RSS delta MB, device peak MB from ``device_peak_fn``).

    The RSS baseline is captured *before* warmup, so the delta reflects all
    allocations triggered by the model + warmup + measurement steps. This
    matters because ``ru_maxrss`` is a lifetime high-water mark — measuring
    after warmup already covers the peak and yields zero delta.
    """
    gc.collect()
    rss_before = _ru_maxrss_mb()
    init_peak = device_peak_fn()
    for _ in range(WARMUP_STEPS):
        step()
    device_peak_fn()  # drop the warm-up window (torch only: XLA's peak cannot be reset)
    tracemalloc.start()
    tracemalloc.reset_peak()
    for _ in range(MEASURE_STEPS):
        step()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    dev_peak = max(init_peak, device_peak_fn())
    rss_after = _ru_maxrss_mb()
    return peak_bytes / (1024 * 1024), max(rss_after - rss_before, 0.0), dev_peak


def _data(B, K, classes, seed=0):
    rng = np.random.default_rng(seed)
    while True:
        yield (
            rng.standard_normal((B, K), dtype=np.float32),
            rng.integers(0, classes, (B,), dtype=np.int64),
        )


# ── per-framework measurement (each runs in its own subprocess) ─────────────

def measure_torch(M, B, K, fw_cfg, device="cpu"):
    import torch
    import torch.nn as tnn

    layer = tnn.Linear(K, M).to(device)
    opt = fw_cfg["optimizer"]
    optim = getattr(torch.optim, opt["name"])(layer.parameters(), **opt.get("kwargs", {}))
    loss_fn = tnn.CrossEntropyLoss()
    it = _data(B, K, classes=M)

    def step():
        x, y = (torch.from_numpy(a).to(device) for a in next(it))
        optim.zero_grad()
        loss_fn(layer(x), y).backward()
        optim.step()

    return _run_with_tracking(step, lambda: _torch_device_peak_mb(device))


def measure_ptorch(M, B, K, fw_cfg, device="cpu"):
    import torch
    import torch.nn.functional as F
    import ptorch.nn.modules as pnn
    import ptorch.optim_static as poptim
    from ptorch import config as ptorch_config

    ptorch_config.config.use_projections = True
    if "linf_chunk_elems" in fw_cfg:
        # null = unbounded: the l_inf solve materialises its full B*K*M intermediates.
        cap = fw_cfg["linf_chunk_elems"]
        ptorch_config.config.linf_chunk_elems = sys.maxsize if cap is None else int(cap)

    layer = pnn.Linear(K, M, norm=fw_cfg.get("norm", "l2")).to(device)
    # p.grad holds projection residuals, which the Projection* optimizers consume.
    opt = fw_cfg["optimizer"]
    optim = getattr(poptim, opt["name"])(layer.parameters(), **opt.get("kwargs", {}))
    loss_fn = getattr(pnn, fw_cfg["loss"])()
    it = _data(B, K, classes=M)

    def step():
        x, y = (torch.from_numpy(a).to(device) for a in next(it))
        y_oh = F.one_hot(y, num_classes=M).float()
        optim.zero_grad()
        loss_fn(layer(x), y_oh).sum().backward()
        optim.step()

    return _run_with_tracking(step, lambda: _torch_device_peak_mb(device))


def measure_pjax(M, B, K, fw_cfg, device="cpu"):
    """Measure peak memory of one pjax projection-based update.

    pjax's projection operator for ``dot`` (``bilinear_proj``) is vmapped twice
    inside ``matmul``, so each scalar bilinear projection runs over a (B, M)
    grid of K-vectors — that's where the O(B*M*K) materialisation comes from.
    To exercise this path we have to drive the model through
    ``Optimizer.update``; using plain ``jax.value_and_grad`` would compile to
    a regular matmul with no 3D blowup. The step is jitted, as in bench_pjax,
    so the measured steps run the compiled update rather than re-tracing it.
    """
    try:
        import jax
        import jax.numpy as jnp
        from pjax import nn as pnn
        from pjax.core import api as papi
        import pjax.optim as poptim
    except ImportError:
        return float("nan"), float("nan"), float("nan")

    class JAXLinear(pnn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.linear = pnn.Linear(in_features, out_features)

        def __call__(self, x):
            return self.linear(x)

    model = JAXLinear(K, M)
    key = jax.random.PRNGKey(0)
    params = model.init(key)
    opt = fw_cfg["optimizer"]
    optimizer = getattr(poptim, opt["name"])(**opt.get("kwargs", {}))
    it = _data(B, K, classes=M)

    @jax.jit
    def update(params, x, y):
        def fun(p):
            return papi.cross_entropy(model.apply(p, x), jax.nn.one_hot(y, M))
        res = optimizer.update(fun, params)
        return res[0] if isinstance(res, (tuple, list)) else res

    def step():
        nonlocal params
        x_np, y_np = next(it)
        params = jax.block_until_ready(update(params, jnp.asarray(x_np), jnp.asarray(y_np)))

    def device_peak_mb():
        stats = jax.devices()[0].memory_stats() if device == "cuda" else None
        return stats["peak_bytes_in_use"] / (1024 * 1024) if stats else float("nan")

    return _run_with_tracking(step, device_peak_mb)


MEASURE_FNS = {
    "torch":  measure_torch,
    "ptorch": measure_ptorch,
    "pjax":   measure_pjax,
}


# ── subprocess isolation ────────────────────────────────────────────────────

NAN3 = (float("nan"),) * 3


def _split(name):
    """'ptorch_linf' -> ('ptorch', {'norm': 'linf'}); 'torch' -> ('torch', {})."""
    base, _, norm = name.partition("_")
    return base, ({"norm": norm} if norm else {})


def _entry(name, M, B, K, fw_cfg, device, ret):
    ret[name] = MEASURE_FNS[_split(name)[0]](M, B, K, fw_cfg, device=device)


def run_isolated(name, M, B, K, fw_cfg, device="cpu"):
    """Returns (tracemalloc_peak_mb, rss_delta_mb, device_peak_mb)."""
    ctx = multiprocessing.get_context("spawn")
    mgr = ctx.Manager()
    ret = mgr.dict()
    p = ctx.Process(target=_entry, args=(name, M, B, K, fw_cfg, device, ret))
    p.start()
    p.join()
    if p.exitcode != 0:
        return NAN3
    return ret.get(name, NAN3)


# ── main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Peak-memory benchmark for one linear-layer projection")
    p.add_argument("--config", required=True, help="experiment YAML")
    cfg = common.load_config(p.parse_args().config)
    device, frameworks = cfg["device"], cfg["frameworks"]
    if device == "cuda":
        # Spawned children inherit this; the module-level setdefault keeps it.
        os.environ["JAX_PLATFORMS"] = "cuda"

    # sweep: in_dim (K swept, B fixed) | batch (B swept, K fixed)
    sweep = cfg.get("sweep", "in_dim")
    if sweep == "batch":
        points = [(B, cfg["in_dim"], cfg["out_dim"]) for B in cfg["batch_sizes"]]
        out_name = "memory_vs_batch.csv"
    else:
        points = [(cfg["batch_size"], K, cfg["out_dim"]) for K in cfg["in_dims"]]
        out_name = "memory_vs_k.csv"

    print(f"sweep={sweep}  (B, K, M) in {points}  frameworks={frameworks}  device={device}")
    print("device peak MB / peak RSS delta MB [tracemalloc peak MB]")
    rows = []
    for B, K, M in points:
        mem = {}
        for fw in frameworks:
            base, extra = _split(fw)
            mem[fw] = run_isolated(fw, M, B, K, {**cfg[base], **extra}, device)
        print(f"B={B:>5} K={K:>6} M={M:>6}  " + "  ".join(
            f"{fw}: {m[2]:.1f} / {m[1]:.1f} [{m[0]:.1f}]" for fw, m in mem.items()))
        row = {"M": M, "B": B, "K": K, "device": device}
        for fw, (tr_mb, rss_mb, dev_mb) in mem.items():
            row.update({f"{fw}_tracemalloc_mb": tr_mb, f"{fw}_rss_mb": rss_mb, f"{fw}_device_mb": dev_mb})
        rows.append(row)

    os.makedirs(cfg["results_dir"], exist_ok=True)
    out = os.path.join(cfg["results_dir"], out_name)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
