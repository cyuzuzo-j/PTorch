"""MLP benchmark: Torch baseline (backprop + cross-entropy).

    python -m experiments.neurips.mlp.bench_torch --config experiments/neurips/mlp/config.yaml

Uses the YAML's `torch:` block (variant name and optimizer). Same tasks,
seeds, data splits and CSV format as bench_ptorch.
"""
import os

import torch
import torch.nn as tnn
import torch.nn.functional as F

from experiments.neurips import common

torch.set_float32_matmul_precision("high")  # same matmul precision as bench_ptorch
EAGER = os.environ.get("TORCHDYNAMO_DISABLE", "0") == "1"  # eager runs get their own CSV: variant + "_eager"


class Loop:
    """common.fit's data and evaluation hooks (shared with bench_ptorch) and `peak`:
    the CUDA allocator peak over the training steps, net of the memory live before
    the model is built or kept by the data module (on CIFAR, every split on the GPU).
    That is what PJAX's XLA peak counts: params, grads, optimizer state, batch and
    step workspace. Evaluation is left out: its loaders re-normalise splits on the GPU."""
    def __init__(self, cfg, task, batch_size, seed, noise, device):
        self.train_iter, self.val_loader, self.test_loader = common.make_data(cfg, task, batch_size, seed, noise)
        self.device, self.cuda = device, device.type == "cuda"
        self.peak = 0 if self.cuda else float("nan")  # bytes
        self.excluded = self.allocated()

    def allocated(self):
        return torch.cuda.memory_allocated(self.device) if self.cuda else 0

    def window(self, fold=True):
        """Fold the allocator peak since the last call into self.peak, then reset it."""
        if self.cuda:
            if fold:
                self.peak = max(self.peak, torch.cuda.max_memory_allocated(self.device) - self.excluded)
            torch.cuda.reset_peak_memory_stats(self.device)

    def tensors(self, x, y):
        return (torch.as_tensor(x, dtype=torch.float32, device=self.device),
                torch.as_tensor(y, dtype=torch.long, device=self.device))

    def next_batch(self):
        self.window()  # closes the previous training step's window
        before = self.allocated()
        x, y = next(self.train_iter)
        self.excluded += self.allocated() - before  # kept by the train stream (CIFAR: a normalised copy)
        return self.tensors(x, y)

    def eval_acc(self, model, loader):
        def predict(x, y):
            x, y = self.tensors(x, y)
            return (model(x).argmax(-1) == y).sum().item(), len(y)
        self.window()  # closes the last training step's window; the evaluation itself is not counted
        model.eval()
        with torch.no_grad():
            acc = common.accuracy(predict, loader)
        model.train()
        self.window(fold=False)
        return acc


class MLP(tnn.Module):
    def __init__(self, task):
        super().__init__()
        layers, last = [], task["in_features"]
        for width in task["hidden"]:
            layers += [tnn.Linear(last, width), tnn.ReLU()]
            last = width
        self.body = tnn.Sequential(*layers)
        self.out = tnn.Linear(last, task["classes"])
        # Same init as ptorch.nn.Linear: Kaiming-normal weights, zero bias.
        for m in self.modules():
            if isinstance(m, tnn.Linear):
                tnn.init.kaiming_normal_(m.weight)
                tnn.init.zeros_(m.bias)

    def forward(self, x):
        return self.out(self.body(x.reshape(x.shape[0], -1)))


def run(cfg, task, batch_size, seed, noise, device, fresh):
    tcfg = cfg["torch"]
    torch.manual_seed(seed)
    loop = Loop(cfg, task, batch_size, seed, noise, device)
    model = MLP(task).to(device)
    opt = tcfg["optimizer"]
    optimizer = getattr(torch.optim, opt["name"])(model.parameters(), **opt.get("kwargs", {}))

    def train_step(batch):
        x, y = batch
        optimizer.zero_grad()
        F.cross_entropy(model(x), y).backward()
        optimizer.step()

    # Plain eager PyTorch: no compiled optimizer, no CUDA graph.
    # Peak memory counts from here; the dataset and evaluation passes are not
    # (see Loop).
    loop.window(fold=False)
    next_batch = loop.next_batch

    timer = common.Timer(device)
    curve, steps, best = common.fit(
        cfg, next_batch, train_step, lambda: loop.eval_acc(model, loop.val_loader),
        snapshot=lambda: {k: t.clone() for k, t in model.state_dict().items()},
        timer=timer, desc=f"{task['name']} | {tcfg['name']} | seed {seed}")
    model.load_state_dict(best)

    meta = {"framework": "torch", "variant": tcfg["name"] + ("_eager" if EAGER else ""), "task": task["name"],
            "optimizer": opt["name"], "label_noise": noise, "batch_size": batch_size, "seed": seed}
    common.save_run(cfg, meta, curve, steps, loop.eval_acc(model, loop.test_loader), timer, fresh,
                    eager=EAGER, peak_mb=loop.peak / 2**20)


def main():
    args = common.parse_args("Torch MLP baseline")
    cfg = common.load_config(args.config, args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for batch_size in cfg["batch_sizes"]:
        for task in cfg["tasks"]:
            for noise in cfg.get("label_noise", [0.0]):
                for seed in common.seeds(cfg):
                    run(cfg, task, batch_size, seed, noise, device, args.fresh)
                    common.free_memory()


if __name__ == "__main__":
    main()
