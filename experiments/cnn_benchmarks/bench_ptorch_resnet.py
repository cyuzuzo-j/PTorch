##################################################
###   Benchmark — ptorch (cyclic projections)  ###
###   ResNet-8 architecture                    ###
##################################################
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))

import gc
import yaml
import torch
import torch.nn.functional as F
import ptorch.optim_static as ptorch_optim_static
import ptorch.nn.modules as pnn
import ptorch.config as ptorch_config
import tqdm, time
import wandb
import torch.nn as tnn
from ptorch.nn.modules import ProjectionModule, HardMarginLoss
from ptorch.core.overrides import apply_overrides
from ptorch.core.ops import process_activation_target

from experiments.shared.data import MNISTDataModule, InfiniteCifarDataModule

apply_overrides()
ptorch_config.use_projections = True

FRAMEWORK = "ptorch_resnet8"
OPTIM_MODULES = vars(ptorch_optim_static)
CFG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

DATASETS = {"MNIST": MNISTDataModule, "CIFAR10": InfiniteCifarDataModule}

# ── Projection-aware global average pooling ──────────────────────────────────

class ProjectedGlobalAvgPool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.mean(dim=[2, 3])

    @staticmethod
    def backward(ctx, z_target):
        x, = ctx.saved_tensors
        N = x.shape[2] * x.shape[3]
        mean_x = x.mean(dim=[2, 3], keepdim=True)
        z_exp = z_target.unsqueeze(-1).unsqueeze(-1)
        correction = (z_exp - mean_x) / (N + 1)
        return process_activation_target(x, x + correction)

# ── Residual addition with independent target propagation ───────────────────

class ResidualAdd(torch.autograd.Function):
    """Addition for residual connections where the two inputs are independent.

    Unlike ProjectedAdd (joint 50/50 split), this sends independent targets:
      skip_target = z_target - out   (full correction flows through skip)
      out_target  = z_target - skip  (residual correction for conv branch)
    """
    @staticmethod
    def forward(ctx, skip, out):
        ctx.save_for_backward(skip, out)
        return skip + out

    @staticmethod
    def backward(ctx, z_target):
        skip, out = ctx.saved_tensors
        return process_activation_target(out, z_target - out), process_activation_target(skip, z_target - skip)

# ── Model ────────────────────────────────────────────────────────────────────

class BasicBlock(tnn.Module):
    def __init__(self, in_channels, out_channels, stride=1, alpha=1.0, g=1.0):
        super().__init__()
        self.conv1 = pnn.Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, alpha=alpha, g=g)
        self.relu1 = pnn.LeakyReLU(0.1)
        self.conv2 = pnn.Conv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1, alpha=alpha, g=g)
        self.relu2 = pnn.LeakyReLU(0.1)

        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = pnn.Conv2D(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, alpha=alpha, g=g)

        with torch.no_grad():
            self.conv2.linear.weight.mul_(0.0)


    def forward(self, x):
        conv_in = x.detach().requires_grad_(True)
        out = self.relu1(self.conv1(conv_in))
        out = self.conv2(out)

        if self.shortcut is not None:
            skip = self.shortcut(x)
        else:
            skip = x

        x = ResidualAdd.apply(skip, out)
        x = self.relu2(x)
        return x


class ResNet8_PTorch(tnn.Module):
    def __init__(self, classes=10, in_channels=3, alpha=1.0, g=1.0):
        super().__init__()
        self.block1 = BasicBlock(in_channels, 64, stride=4, alpha=alpha, g=g)
        #self.block2 = BasicBlock(16, 32, stride=2, alpha=alpha, g=g)
        #self.block3 = BasicBlock(32, 64, stride=2, alpha=alpha, g=g)

        self.head = pnn.Linear(64, classes, norm="inf")
        self.loss = HardMarginLoss()

    def _forward_features(self, x):
        x = self.block1(x)
        #x = self.block2(x)
        #x = self.block3(x)
        x = ProjectedGlobalAvgPool.apply(x)
        return x

    def forward(self, x, y_oh):
        x = self._forward_features(x)
        logits = self.head(x)
        return self.loss(logits, y_oh)

    def forward_eval(self, x):
        x = self._forward_features(x)
        logits = self.head(x)
        return logits


# ── Training ─────────────────────────────────────────────────────────────────

def run(cfg, task_cfg, batch_size, run_number, device):
    seed = cfg["random_seed"]
    run_seed = seed + run_number
    torch.manual_seed(run_seed)

    dataset_name = task_cfg.get("dataset", task_cfg["name"])
    dataset_cls = DATASETS[dataset_name]

    if dataset_name == "CIFAR10":
        ds = dataset_cls(batch_size=batch_size, data_dir="./dataset", seed=run_seed)
    else:
        ds = dataset_cls(batch_size=batch_size, seed=run_seed)

    train_iter = ds.train_iterator()
    val_loader  = ds.val_dataloader()
    test_loader = ds.test_dataloader()

    model = ResNet8_PTorch(classes=task_cfg["classes"], in_channels=task_cfg.get("in_channels", 3)).to(device)

    opt_name   = cfg["ptorch_optimizer"]
    opt_kwargs = cfg.get("ptorch_optimizer_kwargs", {})
    optimizer  = OPTIM_MODULES[opt_name](model.parameters(), **opt_kwargs)

    run_name = f"{cfg.get('experiment_name', 'run')}_{FRAMEWORK}_{task_cfg['name']}_bs{batch_size}_run{run_number}_{opt_name}"
    run_wandb = wandb.init(project="pjax", name=run_name)
    wandb.config.update({
        "framework": FRAMEWORK,
        "task": task_cfg["name"],
        "optimizer": opt_name,
        **{f"opt_{k}": v for k, v in opt_kwargs.items()},
        "batch_size": batch_size,
        "K": cfg.get("K", 1),
        "seed": run_seed,
        "run_number": run_number,
        "max_steps": cfg["max_steps"],
        "eval_every": cfg["eval_every"],
        "patience": cfg["patience"],
        **ptorch_config.snapshot(),
    })

    def eval_acc(loader):
        model.eval()
        accs = []
        with torch.no_grad():
            for xv, yv in loader:
                xv = torch.tensor(xv, dtype=torch.float32, device=device)
                if xv.shape[-1] in [1, 3]:
                    xv = xv.permute(0, 3, 1, 2)
                yv = torch.tensor(yv, dtype=torch.long,  device=device)
                logits = model.forward_eval(xv)
                accs.append((logits.argmax(dim=-1) == yv).float().mean())
        model.train()
        return float(torch.stack(accs).mean())

    best_val_acc, best_state, best_step = 0.0, None, 0
    no_improve = 0
    t0 = time.time()

    K = cfg.get("K", 1)
    step = 0

    with tqdm.tqdm(total=cfg["max_steps"], unit="step") as pbar:
        while step < cfg["max_steps"]:
            x_np, y_np = next(train_iter)
            x_batch = torch.tensor(x_np, dtype=torch.float32, device=device)
            if x_batch.shape[-1] in [1, 3]:
                x_batch = x_batch.permute(0, 3, 1, 2)
            y_batch = torch.tensor(y_np, dtype=torch.long, device=device)
            y_oh    = F.one_hot(y_batch, num_classes=task_cfg["classes"]).float()

            model.train()
            output = model(x_batch, y_oh)

            optimizer.zero_grad()
            output.sum().backward()
            optimizer.step()

            if step % cfg["eval_every"] == 0:
                val_acc = eval_acc(val_loader)
                wandb.log({"val/val_acc": val_acc, "training_time_s": time.time() - t0}, step=step)
                pbar.set_postfix(val_acc=f"{val_acc:.4f}", best=f"{best_val_acc:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc, best_step = val_acc, step
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= cfg["patience"]:
                    print(f"Early stop at step {step}")
                    break

            with torch.no_grad():
                logits = model.forward_eval(x_batch)
            ce_loss   = float(F.cross_entropy(logits, y_batch))
            train_acc = float((logits.argmax(dim=-1) == y_batch).float().mean())
            wandb.log({"train/loss": ce_loss, "train/train_acc": train_acc}, step=step)

            pbar.update(1)
            step += 1

            if no_improve >= cfg["patience"]:
                break

    total_time = time.time() - t0
    if best_state:
        model.load_state_dict(best_state)

    test_acc = eval_acc(test_loader)
    print(f"Test Acc: {test_acc:.4f}  Time: {total_time:.1f}s")
    wandb.log({"test/test_acc": test_acc, "total_training_time_s": total_time}, step=step)
    wandb.finish()


if __name__ == "__main__":
    cfg    = yaml.safe_load(open(CFG_PATH))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for batch_size in cfg["batch_sizes"]:
        for task_cfg in cfg["tasks"]:
            print(f"\n{'='*50}\n{FRAMEWORK} | {task_cfg['name']} | bs={batch_size}")
            for run_number in range(1, cfg["num_runs"] + 1):
                run(cfg, task_cfg, batch_size, run_number, device)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
