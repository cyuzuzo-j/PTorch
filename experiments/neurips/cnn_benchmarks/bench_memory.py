"""Peak GPU memory of one training step of a 1-block CNN in Torch, PTorch and PJAX.

    python -m experiments.neurips.cnn_benchmarks.bench_memory --config experiments/neurips/cnn_benchmarks/memory_config.yaml

Network, identical in all three frameworks:
conv kxk (F filters, 'same' padding) -> ReLU -> 2x2 max-pool -> flatten -> linear head.
ReLU because PJAX has no LeakyReLU; PJAX keeps the conv bias in its ReLU module.

The YAML's `conv_sizes` ([[k, F], ...]) are swept at a fixed `batch_size`. Per image,
PJAX's projection stores one patch vector per (position, filter), ~ positions x patch x F,
the product of the layer's dimensions; PTorch stores the patch matrix plus per-element
solver arrays, ~ positions x (patch + c F), their sum. Each framework trains with the
optimizer and loss of its YAML block, so optimizer state is counted.

Measurement as in mlp/bench_memory.py: every (framework, size) point runs in a fresh
`spawn` subprocess; the column is the framework allocator's device peak
(torch.cuda.max_memory_allocated / JAX peak_bytes_in_use). A point that runs out of
GPU memory is NaN in the CSV.
"""
import argparse
import multiprocessing
import os

import numpy as np
import pandas as pd

from experiments.neurips import common
from experiments.neurips.mlp.bench_memory import NAN3, _run_with_tracking, _torch_device_peak_mb


def _images(B, net, seed=0):
    rng = np.random.default_rng(seed)
    S, C = net["size"], net["in_channels"]
    while True:
        yield (rng.standard_normal((B, S, S, C), dtype=np.float32),
               rng.integers(0, net["classes"], (B,), dtype=np.int64))


def _cnn(nn, conv, act, pool, linear, net):
    k, F, S = net["kernel_size"], net["filters"], net["size"]
    return nn.Sequential(conv(net["in_channels"], F, k, k // 2), act(), pool(2), nn.Flatten(),
                         linear((S // 2) ** 2 * F, net["classes"]))


def measure_torch(net, B, fw_cfg, device):
    import torch
    import torch.nn as tnn

    model = _cnn(tnn, lambda i, o, k, p: tnn.Conv2d(i, o, k, padding=p), tnn.ReLU, tnn.MaxPool2d, tnn.Linear,
                 net).to(device)
    opt = fw_cfg["optimizer"]
    optim = getattr(torch.optim, opt["name"])(model.parameters(), **opt.get("kwargs", {}))
    it = _images(B, net)

    def step():
        x, y = (torch.from_numpy(a).to(device) for a in next(it))
        optim.zero_grad()
        tnn.functional.cross_entropy(model(x.permute(0, 3, 1, 2)), y).backward()
        optim.step()

    return _run_with_tracking(step, lambda: _torch_device_peak_mb(device))


def measure_ptorch(net, B, fw_cfg, device):
    import torch
    import torch.nn as tnn
    import torch.nn.functional as F
    import ptorch.nn.modules as pnn
    import ptorch.optim_static as poptim
    from ptorch.config import config as ptorch_config

    ptorch_config.use_projections = True
    model = _cnn(tnn, lambda i, o, k, p: pnn.Conv2D(i, o, kernel_size=k, padding=p), pnn.ReLU, pnn.MaxPool2d,
                 lambda i, o: pnn.Linear(i, o, norm=fw_cfg.get("head_norm", "l2")), net).to(device)
    opt = fw_cfg["optimizer"]
    optim = getattr(poptim, opt["name"])(model.parameters(), **opt.get("kwargs", {}))
    loss_fn = getattr(pnn, fw_cfg["loss"])()
    it = _images(B, net)

    def step():
        x, y = (torch.from_numpy(a).to(device) for a in next(it))
        optim.zero_grad()
        loss_fn(model(x.permute(0, 3, 1, 2)), F.one_hot(y, num_classes=net["classes"]).float()).sum().backward()
        optim.step()

    return _run_with_tracking(step, lambda: _torch_device_peak_mb(device))


def measure_pjax(net, B, fw_cfg, device):
    """PJAX's own Conv2D -> ReLU(bias) -> 2x2 max (reshape + max) -> flatten -> Linear, driven through
    Optimizer.update (jitted) as in mlp/bench_memory.py. NaN if pjax / JAX are not installed."""
    try:
        import jax
        import jax.numpy as jnp
        import pjax
        from pjax import nn as pjnn
        from pjax.core import api as papi
        import pjax.optim as pjoptim
    except ImportError:
        return NAN3
    k, C, S, classes = net["kernel_size"], net["filters"], net["size"], net["classes"]

    class CNN(pjnn.Module):
        def __init__(self):
            super().__init__()
            self.conv = pjnn.Conv2D(net["in_channels"], C, (k, k), (1, 1), "SAME")
            self.relu = pjnn.ReLU(C)
            self.out = pjnn.Linear((S // 2) ** 2 * C, classes)

        def __call__(self, x):
            x = self.relu(self.conv(x))                                    # (B, S, S, C)
            x = pjax.max(pjax.reshape(x, (x.shape[0], S // 2, 2, S // 2, 2, C)), axis=(2, 4))
            return self.out(pjax.reshape(x, (x.shape[0], -1)))

    model = CNN()
    params = model.init(jax.random.PRNGKey(0))
    opt = fw_cfg["optimizer"]
    optimizer = getattr(pjoptim, opt["name"])(**opt.get("kwargs", {}))
    it = _images(B, net)

    @jax.jit
    def update(params, x, y):
        res = optimizer.update(lambda p: papi.cross_entropy(model.apply(p, x), jax.nn.one_hot(y, classes)), params)
        return res[0] if isinstance(res, (tuple, list)) else res

    def step():
        nonlocal params
        x, y = next(it)
        params = jax.block_until_ready(update(params, jnp.asarray(x), jnp.asarray(y)))

    def device_peak_mb():
        stats = jax.devices()[0].memory_stats() if device == "cuda" else None
        return stats["peak_bytes_in_use"] / (1024 * 1024) if stats else float("nan")

    return _run_with_tracking(step, device_peak_mb)


MEASURE_FNS = {"torch": measure_torch, "ptorch": measure_ptorch, "pjax": measure_pjax}


def _entry(name, net, B, fw_cfg, device, ret):
    ret[name] = MEASURE_FNS[name](net, B, fw_cfg, device)


def run_isolated(name, net, B, fw_cfg, device):
    """(tracemalloc peak MB, RSS delta MB, device peak MB) in a fresh process; NaN if it fails (e.g. OOM)."""
    ctx = multiprocessing.get_context("spawn")
    ret = ctx.Manager().dict()
    p = ctx.Process(target=_entry, args=(name, net, B, fw_cfg, device, ret))
    p.start()
    p.join()
    return ret.get(name, NAN3) if p.exitcode == 0 else NAN3


def main():
    p = argparse.ArgumentParser(description="Peak-memory benchmark of a 1-block CNN")
    p.add_argument("--config", required=True, help="experiment YAML")
    cfg = common.load_config(p.parse_args().config)
    B, device, frameworks = cfg["batch_size"], cfg["device"], cfg["frameworks"]
    if device == "cuda":
        os.environ["JAX_PLATFORMS"] = "cuda"  # spawned children inherit it

    rows = []
    for k, f in cfg["conv_sizes"]:
        net = {**cfg["network"], "kernel_size": k, "filters": f}
        mem = {fw: run_isolated(fw, net, B, cfg[fw], device) for fw in frameworks}
        print(f"conv {k}x{k}, {f} filters, batch {B}:  " + "  ".join(f"{fw} {m[2]:.1f} MB" for fw, m in mem.items()))
        row = {"B": B, "kernel": k, "filters": f, "patch": net["in_channels"] * k * k,
               "conv_params": net["in_channels"] * k * k * f, "device": device}
        for fw, (tr_mb, rss_mb, dev_mb) in mem.items():
            row.update({f"{fw}_tracemalloc_mb": tr_mb, f"{fw}_rss_mb": rss_mb, f"{fw}_device_mb": dev_mb})
        rows.append(row)

    os.makedirs(cfg["results_dir"], exist_ok=True)
    out = os.path.join(cfg["results_dir"], "memory_vs_conv.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
