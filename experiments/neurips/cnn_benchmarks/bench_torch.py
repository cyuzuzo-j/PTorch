"""CNN benchmark: Torch baseline (backprop + cross-entropy).

    python -m experiments.neurips.cnn_benchmarks.bench_torch --config experiments/neurips/cnn_benchmarks/resnet8_config.yaml

Uses the YAML's `torch:` block. Same tasks, seeds, data splits and CSV format
as bench_ptorch.
"""
import torch
import torch.nn.functional as F

from experiments.neurips import common
from experiments.neurips.cnn_benchmarks.models import ResNet8_Torch, to_nchw
from ptorch.config import config as ptorch_config

# models.py imports ptorch, whose op patches (e.g. Tensor.__add__) project instead of
# differentiating while use_projections is on, the library default. The Torch baseline
# trains with plain autograd.
ptorch_config.use_projections = False


def run(cfg, task, batch_size, seed, device, fresh):
    tcfg = cfg["torch"]
    common.set_precision(cfg)
    torch.manual_seed(seed)
    train_iter, val_loader, test_loader = common.make_data(cfg, task, batch_size, seed)
    model = ResNet8_Torch(task["classes"], task["in_channels"], tcfg.get("widths", [64, 128, 256]),
                          act=tcfg.get("act", "leaky")).to(device)
    opt = tcfg["optimizer"]
    optimizer = getattr(torch.optim, opt["name"])(model.parameters(), **opt.get("kwargs", {}))

    def next_batch():
        x, y = next(train_iter)
        return to_nchw(x, device), torch.as_tensor(y, dtype=torch.long, device=device)

    def train_step(batch):
        x, y = batch
        optimizer.zero_grad()
        F.cross_entropy(model(x), y).backward()
        optimizer.step()

    def predict(x, y):
        y = torch.as_tensor(y, dtype=torch.long, device=device)
        return (model(to_nchw(x, device)).argmax(-1) == y).sum().item(), len(y)

    def eval_acc(loader):
        model.eval()
        with torch.no_grad():
            acc = common.accuracy(predict, loader)
        model.train()
        return acc

    on_best, on_regression = common.lr_regression_hooks(cfg, model, optimizer)  # eager: a float lr is fine
    timer = common.Timer(device)
    curve, steps, best = common.fit(
        cfg, next_batch, train_step, lambda: eval_acc(val_loader),
        snapshot=lambda: {k: t.clone() for k, t in model.state_dict().items()},
        timer=timer, desc=f"{task['name']} | {tcfg['name']} | seed {seed}",
        on_best=on_best, on_regression=on_regression)
    model.load_state_dict(best)

    meta = {"framework": "torch", "variant": tcfg["name"], "task": task["name"],
            "optimizer": opt["name"], "batch_size": batch_size, "seed": seed,
            "model": "resnet8", "widths": "-".join(map(str, tcfg.get("widths", [64, 128, 256]))),
            "act": tcfg.get("act", "leaky")}
    common.add_schedule_meta(cfg, meta)
    common.save_run(cfg, meta, curve, steps, eval_acc(test_loader), timer, fresh)


def main():
    args = common.parse_args("Torch CNN baseline")
    cfg = common.load_config(args.config, args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for batch_size in cfg["batch_sizes"]:
        for task in cfg["tasks"]:
            for seed in common.seeds(cfg):
                run(cfg, task, batch_size, seed, device, args.fresh)
                common.free_memory()


if __name__ == "__main__":
    main()
