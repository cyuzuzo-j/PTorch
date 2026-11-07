"""MLP benchmark: PTorch (cyclic projections).

    python -m experiments.neurips.mlp.bench_ptorch --config experiments/neurips/mlp/config.yaml

Runs every variant in the YAML's `ptorch.variants` on every task, batch size
and label-noise level, `num_runs` seeds each. See common.py for the data
protocol and the CSV format.
"""
import copy
import os

# PTorch always runs reduce-overhead: the hot-path compiled impls replay as
# CUDA graphs (ptorch/core/ops.py reads these at import).
os.environ["PTORCH_COMPILE_MODE"] = "reduce-overhead"
os.environ["PTORCH_COMPILE_DYNAMIC"] = "0"

import torch
import torch.nn as tnn
import torch.nn.functional as F

import ptorch.nn.modules as pnn
import ptorch.optim_static as poptim
from ptorch.config import config as ptorch_config
from experiments.neurips import common
from experiments.neurips.mlp.bench_torch import EAGER, Loop  # also sets the "high" matmul precision
from experiments.neurips.mlp.unfused_linear import UnfusedLinear

# PTORCH_COMPILE_MODE=reduce-overhead (see ptorch/core/ops.py) needs one
# cudagraph-trees step per forward+backward.
mark_step = torch.compiler.cudagraph_mark_step_begin if os.environ.get("PTORCH_COMPILE_MODE") else lambda: None
ptorch_config.use_projections = True
ptorch_config.update("cross_entropy_loss_value", False)  # no run reads the loss value
# Each new optimizer recompiles its step, and past the limit dynamo falls back to eager.
torch._dynamo.config.recompile_limit = 16

# Every variant starts from these; the YAML's `ptorch.defaults` and then the
# variant entry itself override them.
VARIANT_DEFAULTS = {
    "norm": "l2",              # projection norm of the hidden linear layers: l2 | linf
    "head_norm": None,         # norm of the output layer (None = norm); l2 + linf = hybrid
    "impl": "fused",           # fused | unfused (ablation row R1, l2 only)
    "num_iters": None,         # Newton steps per bilinear solve (None = library default)
    "optimizer": {"name": "ProjectionMuon", "kwargs": {}},
    "loss": "CrossEntropy",    # any loss class in ptorch.nn.modules
    "passes": 1,               # projection sweeps per batch (K)
    "init": "he",              # weight init: he (library default) | normal (std 0.01) | orthogonal (x sqrt 2)
    "max_steps": None,         # per-variant step budget (equal-compute K sweep)
    # ptorch.config fields, set before each run:
    "use_muon_activations": False,
    "td_lambda": 0.0,          # floor(λ) heuristic strength (0 = off)
    "td_mode": "norm",
    "td_scope": "global",
    "td_eps_lin": 1e-2,
    "td_deflate": False,
    "cross_entropy_lambda": 10.0,  # proximal strength λ of the CE loss
    "cross_entropy_num_steps": 1,  # inner steps of the proximal CE solve
    "projection_alpha_auto": False,  # rescale alpha per l2 solve by mean(qa)/mean(qb)
    "linf_num_iters": 1,           # Newton steps of each l_inf radius solve
    "linf_warm_start": False,      # start the l_inf radius from the previous batch's
    "linf_legacy_newton": False,   # the pre-Oct-25 l_inf Newton step
}
PTORCH_CONFIG_FIELDS = ["use_muon_activations", "td_lambda", "td_mode", "td_scope",
                        "td_eps_lin", "td_deflate", "cross_entropy_lambda", "cross_entropy_num_steps",
                        "projection_alpha_auto", "linf_num_iters", "linf_warm_start",
                        "linf_legacy_newton"]


class MLP(tnn.Module):
    def __init__(self, task, norm, head_norm, impl, num_iters):
        super().__init__()

        def linear(n_in, n_out, layer_norm):
            if impl == "unfused":
                assert layer_norm == "l2", "the unfused reference solve is l2 only"
                return UnfusedLinear(n_in, n_out)
            return pnn.Linear(n_in, n_out, norm=layer_norm, num_iters=num_iters)

        layers, last = [], task["in_features"]
        for width in task["hidden"]:
            layers += [linear(last, width, norm), pnn.ReLU(norm=norm)]
            last = width
        self.body = tnn.Sequential(*layers)
        self.out = linear(last, task["classes"], head_norm or norm)

    def forward(self, x):
        return self.out(self.body(x.reshape(x.shape[0], -1)))


def init_weights(model, init):
    for m in model.modules():
        if not isinstance(m, (pnn.Linear, UnfusedLinear)) or init == "he":
            continue
        if init == "normal":
            tnn.init.normal_(m.weight, std=0.01)
        elif init == "orthogonal":
            tnn.init.orthogonal_(m.weight, gain=2 ** 0.5)
        else:
            raise ValueError(f"unknown init {init!r}")


def build_variants(cfg):
    defaults = {**VARIANT_DEFAULTS, **cfg["ptorch"].get("defaults", {})}
    for entry in [cfg["ptorch"].get("defaults", {}), *cfg["ptorch"]["variants"]]:  # typos are errors
        unknown = set(entry) - set(VARIANT_DEFAULTS) - {"name"}
        if unknown:
            raise ValueError(f"variant {entry.get('name', 'defaults')}: unknown keys {sorted(unknown)}")
    return [{**defaults, **entry} for entry in cfg["ptorch"]["variants"]]


def run(cfg, task, batch_size, seed, noise, v, device, fresh):
    torch.manual_seed(seed)
    loop = Loop(cfg, task, batch_size, seed, noise, device)
    for key in PTORCH_CONFIG_FIELDS:
        ptorch_config.update(key, v[key])

    model = MLP(task, v["norm"], v["head_norm"], v["impl"], v["num_iters"]).to(device)
    init_weights(model, v["init"])
    loss = getattr(pnn, v["loss"])()
    make_opt = lambda m: getattr(poptim, v["optimizer"]["name"])(m.parameters(), **v["optimizer"].get("kwargs", {}))
    optimizer = make_opt(model)

    def one_hot(y):
        return F.one_hot(y, num_classes=task["classes"]).float()

    def train_step(batch):
        x, y = batch
        for _ in range(v["passes"]):
            mark_step()
            value = loss(model(x), one_hot(y))
            # Skip the sum where nothing reads it: CE is scalar already, and HardMargin
            # ignores the target its backward receives.
            if value.ndim == 0:
                value.backward()
            elif isinstance(loss, pnn.HardMarginLoss):
                value.backward(value.detach())
            else:
                value.sum().backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=False)  # keeps grad buffers fixed for the cudagraphs

    # torch.compile warm-up on a throwaway copy, so compile time is reported
    # separately and kept off the training clock.
    t0 = common.sync_time(device)
    warm = copy.deepcopy(model)
    warm_opt = make_opt(warm)
    x0 = torch.zeros(batch_size, task["in_features"], device=device)
    y0 = one_hot(torch.zeros(batch_size, dtype=torch.long, device=device))
    for _ in range(2):
        mark_step()
        loss(warm(x0), y0).sum().backward()
        warm_opt.step()
        warm_opt.zero_grad()
    with torch.no_grad():
        warm.eval()
        warm(x0)
    del warm, warm_opt
    # Compile the optimizer step; the first real batch is the warm-up input and
    # is still trained on.
    # Peak memory counts from here (inaccurate under reduce-overhead, see save_run below);
    # the dataset and evaluation passes are not (see Loop).
    loop.window(fold=False)
    first = [loop.next_batch()]
    train_step = common.compile_step(train_step, model, optimizer, first[0])
    next_batch = lambda: first.pop() if first else loop.next_batch()
    compile_s = common.sync_time(device) - t0

    timer = common.Timer(device)
    curve, steps, best = common.fit(
        cfg, next_batch, train_step, lambda: loop.eval_acc(model, loop.val_loader),
        snapshot=lambda: {k: t.clone() for k, t in model.state_dict().items()},
        timer=timer, desc=f"{task['name']} | {v['name']} | seed {seed}",
        max_steps=v["max_steps"])
    model.load_state_dict(best)

    meta = {"framework": "ptorch", "variant": v["name"] + ("_eager" if EAGER else ""),
            "task": task["name"], "norm": v["norm"], "head_norm": v["head_norm"] or v["norm"],
            "optimizer": v["optimizer"]["name"], "loss": v["loss"], "td_lambda": v["td_lambda"],
            "passes": v["passes"], "init": v["init"], "label_noise": noise, "batch_size": batch_size, "seed": seed}
    # peak_mb is NOT accurate under reduce-overhead (always on here): replayed
    # cudagraphs reuse memory recorded in their private pools without going
    # through the allocator stats, so this under-reports (l_inf: ~86 vs ~603 MB)
    # and varies by seed. Use bench_memory for memory numbers.
    common.save_run(cfg, meta, curve, steps, loop.eval_acc(model, loop.test_loader), timer, fresh,
                    compile_s=compile_s, eager=EAGER, peak_mb=loop.peak / 2**20)


def main():
    args = common.parse_args("PTorch MLP benchmark", variants=True)
    cfg = common.load_config(args.config, args)
    variants = build_variants(cfg)
    if args.variants:
        variants = common.select(variants, args.variants)
    if args.max_steps is not None:  # smoke test: the cap also beats per-variant budgets
        for v in variants:
            v["max_steps"] = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for batch_size in cfg["batch_sizes"]:
        for task in cfg["tasks"]:
            for v in variants:
                for noise in cfg.get("label_noise", [0.0]):
                    for seed in common.seeds(cfg):
                        run(cfg, task, batch_size, seed, noise, v, device, args.fresh)
                        common.free_memory()


if __name__ == "__main__":
    main()
