"""MLP benchmark: PJAX (Douglas-Rachford projections, Bergmeister et al. 2025).

    PYTHONPATH=src:third_party/pjax:. python -m experiments.neurips.mlp.bench_pjax \
        --config experiments/neurips/mlp/config.yaml

Uses the YAML's `pjax:` block (variant name and optimizer). Needs `pjax`
(not on PyPI): the third_party/pjax submodule, github.com/AndreasBergmeister/pjax. Same tasks, seeds, data splits and CSV
format as bench_ptorch.
"""
import itertools
import multiprocessing
import time

import jax
import jax.numpy as jnp
import pjax
import pjax.nn as nn
import pjax.optim as optim
from pjax.core.api import cross_entropy

from experiments.neurips import common

reshape = getattr(pjax, "reshape", jnp.reshape)  # pjax's own reshape when it has one


class MLP(nn.Module):
    def __init__(self, task):
        super().__init__()
        last = task["in_features"]
        for i, width in enumerate(task["hidden"]):
            setattr(self, f"linear_{i}", nn.Linear(last, width))
            setattr(self, f"relu_{i}", nn.ReLU(width))
            last = width
        self.n_hidden = len(task["hidden"])
        self.out = nn.Linear(last, task["classes"])

    def __call__(self, x):
        x = reshape(x, (x.shape[0], -1))
        for i in range(self.n_hidden):
            x = getattr(self, f"relu_{i}")(getattr(self, f"linear_{i}")(x))
        return self.out(x)


def run(cfg, task, batch_size, seed, noise, fresh):
    pcfg = cfg["pjax"]
    train_iter, val_loader, test_loader = common.make_data(cfg, task, batch_size, seed, noise)
    model = MLP(task)
    params = model.init(jax.random.split(jax.random.key(seed))[0])
    opt = pcfg["optimizer"]
    optimizer = getattr(optim, opt["name"])(**opt.get("kwargs", {}))

    @jax.jit
    def step_fn(params, x, y):
        def objective(p):
            pred = model.apply(p, x)
            return cross_entropy(pred, jax.nn.one_hot(y, pred.shape[-1]))
        res = optimizer.update(objective, params)
        return res[0] if isinstance(res, (tuple, list)) else res

    @jax.jit
    def n_correct(params, x, y):
        pred = model.apply(params, x)
        pred = getattr(pred, "value", pred)
        return jnp.sum(jnp.argmax(pred, axis=-1) == y)

    def eval_acc(p, loader):
        return common.accuracy(lambda x, y: (int(n_correct(p, jnp.array(x), jnp.array(y))), len(y)), loader)

    # XLA compile on the first batch, pushed back so training still sees it; off the clock.
    train_iter = itertools.chain([first := next(train_iter)], train_iter)
    t0 = time.perf_counter()
    x0, y0 = jnp.array(first[0]), jnp.array(first[1])
    jax.block_until_ready(step_fn(params, x0, y0))
    n_correct(params, x0, y0).block_until_ready()
    compile_s = time.perf_counter() - t0

    state = {"params": params}

    def next_batch():
        x, y = next(train_iter)
        return jnp.array(x), jnp.array(y)

    def train_step(batch):
        state["params"] = jax.block_until_ready(step_fn(state["params"], *batch))

    timer = common.Timer()
    curve, steps, best = common.fit(
        cfg, next_batch, train_step, lambda: eval_acc(state["params"], val_loader),
        snapshot=lambda: state["params"],  # jax arrays are immutable: no copy needed
        timer=timer, desc=f"{task['name']} | {pcfg['name']} | seed {seed}")

    stats = jax.devices()[0].memory_stats() or {}
    meta = {"framework": "pjax", "variant": pcfg["name"], "task": task["name"],
            "optimizer": opt["name"], "label_noise": noise, "batch_size": batch_size, "seed": seed}
    common.save_run(cfg, meta, curve, steps, eval_acc(best, test_loader), timer, fresh,
                    compile_s=compile_s,
                    peak_mb=stats.get("peak_bytes_in_use", float("nan")) / 2**20)


def run_isolated(*args):
    """Run one training run in a fresh process: XLA's peak_bytes_in_use is a
    process-lifetime high-water mark that cannot be reset, so this is the only
    way to give peak_mb the per-run window torch's reset_peak_memory_stats gives
    the other benches. The compile step has the training shapes, so it does not
    raise the peak."""
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=run, args=args)
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"pjax run failed (exit code {p.exitcode})")


def main():
    args = common.parse_args("PJAX MLP benchmark")
    cfg = common.load_config(args.config, args)
    cleared = set()  # --fresh clears each task's CSV once; the child processes can't track that
    for batch_size in cfg["batch_sizes"]:
        for task in cfg["tasks"]:
            for noise in cfg.get("label_noise", [0.0]):
                for seed in common.seeds(cfg):
                    fresh = args.fresh and task["name"] not in cleared
                    cleared.add(task["name"])
                    run_isolated(cfg, task, batch_size, seed, noise, fresh)


if __name__ == "__main__":
    main()
