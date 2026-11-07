"""CNN benchmark: PTorch (cyclic projections).

    python -m experiments.neurips.cnn_benchmarks.bench_ptorch --config experiments/neurips/cnn_benchmarks/resnet8_config.yaml

Runs every variant in the YAML's `ptorch.variants` on every task and batch
size, `num_runs` seeds each. See experiments/neurips/common.py for the data
protocol and the CSV format.
"""
import copy
import os

# PTorch always runs reduce-overhead: the hot-path compiled impls replay as
# CUDA graphs (ptorch/core/ops.py reads these at import).
os.environ["PTORCH_COMPILE_MODE"] = "reduce-overhead"
os.environ["PTORCH_COMPILE_DYNAMIC"] = "0"

import torch
import torch.nn.functional as F

import ptorch.nn.modules as pnn
import ptorch.optim_static as poptim
from ptorch.config import config as ptorch_config
from experiments.neurips import common
from experiments.neurips.cnn_benchmarks.models import ResidualCNN_PTorch, ResNet8_PTorch, to_nchw

# PTORCH_COMPILE_MODE=reduce-overhead (see ptorch/core/ops.py) needs one
# cudagraph-trees step per forward+backward.
mark_step = torch.compiler.cudagraph_mark_step_begin if os.environ.get("PTORCH_COMPILE_MODE") else lambda: None
ptorch_config.use_projections = True

VARIANT_DEFAULTS = {
    "model": "resnet8",        # resnet8 | residual
    "conv_features": 32,       # channels of the first block (residual)
    "widths": [64, 128, 256],  # channels of the three basic blocks (resnet8)
    "branch_mode": "delta_sum",  # skip/main-path target combine at each block input (resnet8)
    "act": "leaky",            # leaky (LeakyReLU 0.1) | relu (resnet8)
    "head_norm": "l2",         # projection norm of the linear head (resnet8)
    "optimizer": {"name": "ProjectionAdam", "kwargs": {}},
    "loss": "CrossEntropy",
}


def build_model(task, v):
    if v["model"] == "resnet8":
        return ResNet8_PTorch(task["classes"], task["in_channels"], v["widths"], head_norm=v["head_norm"],
                              branch_mode=v["branch_mode"], act=v["act"])
    if v["model"] == "residual":
        return ResidualCNN_PTorch(task["classes"], task["in_channels"], v["conv_features"])
    raise ValueError(f"unknown model {v['model']!r}")


def build_variants(cfg):
    defaults = {**VARIANT_DEFAULTS, **cfg["ptorch"].get("defaults", {})}
    for entry in [cfg["ptorch"].get("defaults", {}), *cfg["ptorch"]["variants"]]:  # typos are errors
        unknown = set(entry) - set(VARIANT_DEFAULTS) - {"name"}
        if unknown:
            raise ValueError(f"variant {entry.get('name', 'defaults')}: unknown keys {sorted(unknown)}")
    return [{**defaults, **entry} for entry in cfg["ptorch"]["variants"]]


def run(cfg, task, batch_size, seed, v, device, fresh):
    common.set_precision(cfg)
    torch.manual_seed(seed)
    train_iter, val_loader, test_loader = common.make_data(cfg, task, batch_size, seed)
    model = build_model(task, v).to(device)
    loss = getattr(pnn, v["loss"])()
    opt = v["optimizer"]
    kwargs = opt.get("kwargs", {})
    if cfg.get("lr_on_regression"):  # lr as a tensor: lr changes need no recompile
        if opt["name"] != "ProjectionAdam":
            raise ValueError(f"lr_on_regression supports ProjectionAdam only, not {opt['name']}")
        kwargs = common.tensor_lr(kwargs, device)
    make_opt = lambda m: getattr(poptim, opt["name"])(m.parameters(), **kwargs)
    optimizer = make_opt(model)

    def next_batch():
        x, y = next(train_iter)
        y = torch.as_tensor(y, dtype=torch.long, device=device)
        return to_nchw(x, device), F.one_hot(y, num_classes=task["classes"]).float()

    def train_step(batch, model=model, optimizer=optimizer):
        x, y = batch
        mark_step()
        loss(model(x), y).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=False)  # keeps grad buffers fixed for the cudagraphs

    def predict(x, y):
        y = torch.as_tensor(y, dtype=torch.long, device=device)
        return (model(to_nchw(x, device)).argmax(-1) == y).sum().item(), len(y)

    def eval_acc(loader):
        model.eval()
        with torch.no_grad():
            acc = common.accuracy(predict, loader)
        model.train()
        return acc

    # Inductor warm-up on a throwaway copy with training-shaped zeros (H, W, C taken
    # from a val batch; the train stream is untouched), so compile time is reported
    # separately and kept off the training clock.
    t0 = common.sync_time(device)
    warm, hwc = copy.deepcopy(model), next(iter(val_loader))[0].shape[1:]
    zeros = (to_nchw(torch.zeros(batch_size, *hwc), device), torch.zeros(batch_size, task["classes"], device=device))
    warm_opt = make_opt(warm)
    for _ in range(2):
        train_step(zeros, warm, warm_opt)
    with torch.no_grad():
        warm.eval()
        warm(zeros[0])
    del warm, warm_opt
    # Compiled optimizer step, warmed on the first real batch, which is pushed
    # back so training still sees it.
    first = [next_batch()]
    common.compile_step(train_step, model, optimizer, first[0])
    stream = next_batch
    next_batch = lambda: first.pop() if first else stream()
    compile_s = common.sync_time(device) - t0

    on_best, on_regression = common.lr_regression_hooks(cfg, model, optimizer)
    timer = common.Timer(device)
    curve, steps, best = common.fit(
        cfg, next_batch, train_step, lambda: eval_acc(val_loader),
        snapshot=lambda: {k: t.clone() for k, t in model.state_dict().items()},
        timer=timer, desc=f"{task['name']} | {v['name']} | seed {seed}",
        on_best=on_best, on_regression=on_regression)
    model.load_state_dict(best)

    meta = {"framework": "ptorch", "variant": v["name"], "task": task["name"],
            "model": v["model"], "head_norm": v["head_norm"], "optimizer": opt["name"],
            "loss": v["loss"], "batch_size": batch_size, "seed": seed}
    if v["model"] == "resnet8":
        meta.update(widths="-".join(map(str, v["widths"])), branch_mode=v["branch_mode"], act=v["act"])
    common.add_schedule_meta(cfg, meta)
    common.save_run(cfg, meta, curve, steps, eval_acc(test_loader), timer, fresh,
                    compile_s=compile_s)


def main():
    args = common.parse_args("PTorch CNN benchmark", variants=True)
    cfg = common.load_config(args.config, args)
    variants = build_variants(cfg)
    if args.variants:
        variants = common.select(variants, args.variants)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for batch_size in cfg["batch_sizes"]:
        for task in cfg["tasks"]:
            for v in variants:
                for seed in common.seeds(cfg):
                    run(cfg, task, batch_size, seed, v, device, args.fresh)
                    common.free_memory()


if __name__ == "__main__":
    main()
