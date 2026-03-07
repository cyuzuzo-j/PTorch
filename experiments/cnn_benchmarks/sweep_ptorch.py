#!/usr/bin/env python3
"""
WandB Sweep Agent — PTorch CNN on CIFAR-10

Usage:
  # 1. Create the sweep (once):
  wandb sweep experiments/cnn_benchmarks/sweep_ptorch.yaml

  # 2. Start agent(s) (can run multiple in parallel):
  wandb agent <SWEEP_ID>
"""
import sys, os, json, gc, time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))

import torch
import torch.nn.functional as F
from ptorch.core.ops import MarginLossProjection
import ptorch.optim_static as ptorch_optim_static
import ptorch.config as ptorch_config
import tqdm
import wandb

from experiments.cnn_benchmarks.models import CNN_PTorch
from experiments.shared.data import InfiniteCifarDataModule

FRAMEWORK = "ptorch_parr"
OPTIM_MODULES = {
    "AlternatingProjections":       ptorch_optim_static.AlternatingProjections,
    "ProjectionSGD":                ptorch_optim_static.ProjectionSGD,
    "AlternatingProjectionsMomentum": ptorch_optim_static.AlternatingProjectionsMomentum,
    "ProjectionAdam":               ptorch_optim_static.ProjectionAdam,
    "ProjectionAdagrad":            ptorch_optim_static.ProjectionAdagrad,
    "ProjectionAdadelta":           ptorch_optim_static.ProjectionAdadelta,
}

# Optimizers that accept a learning rate parameter
LR_OPTIMIZERS = {
    "ProjectionSGD", "AlternatingProjectionsMomentum",
    "ProjectionAdam", "ProjectionAdagrad", "ProjectionAdadelta",
}


def train():
    """Single training run driven by wandb.config."""
    run = wandb.init()
    config = wandb.config

    # ── Parse config ─────────────────────────────
    opt_name    = config.optimizer
    lr          = config.lr
    alpha       = config.alpha
    g           = config.g
    batch_size  = config.batch_size
    conv_widths = json.loads(config.architecture)  # "[4, 8, 8]" → [4, 8, 8]
    max_steps   = config.max_steps
    eval_every  = config.eval_every
    patience    = config.patience
    seed        = config.seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    # ── Data ─────────────────────────────────────
    ds = InfiniteCifarDataModule(batch_size=batch_size, data_dir="./dataset", seed=seed)
    train_iter  = ds.train_iterator()
    val_loader  = ds.val_dataloader()
    test_loader = ds.test_dataloader()

    # ── Model ────────────────────────────────────
    model = CNN_PTorch(classes=10, in_channels=3, conv_widths=conv_widths, alpha=alpha, g=g).to(device)

    # ── Optimizer ────────────────────────────────
    opt_cls = OPTIM_MODULES[opt_name]
    if opt_name in LR_OPTIMIZERS:
        optimizer = opt_cls(model.parameters(), lr=lr)
    else:
        # AlternatingProjections takes no kwargs
        optimizer = opt_cls(model.parameters())

    # ── Log full config ──────────────────────────
    wandb.config.update({
        "framework": FRAMEWORK,
        "task": "CIFAR10",
        "conv_widths": conv_widths,
        "alpha": alpha,
        "g": g,
        "num_params": sum(p.numel() for p in model.parameters()),
        **ptorch_config.snapshot(),
    }, allow_val_change=True)

    # ── Training step ────────────────────────────
    def step_fn(x, y):
        logits = model(x)
        y_oh = F.one_hot(y.long(), num_classes=10).float()
        projected = MarginLossProjection.apply(logits, y_oh)
        optimizer.zero_grad()
        projected.sum().backward()
        loss = F.cross_entropy(logits.detach(), y.long())
        optimizer.step()
        return loss

    def eval_fn(x, y):
        with torch.no_grad():
            return (model(x).argmax(dim=-1) == y).float().mean()

    # ── Training loop ────────────────────────────
    best_val_acc, best_state, best_step = 0.0, None, 0
    no_improve = 0
    step = 0
    t0 = time.time()

    with tqdm.tqdm(total=max_steps, unit="step") as pbar:
        while step < max_steps:
            # ── Evaluate ─────────────────────────
            if step % eval_every == 0:
                model.eval()
                accs = []
                for x, y in val_loader:
                    x_t = torch.tensor(x, dtype=torch.float32, device=device)
                    if x_t.shape[-1] in [1, 3]:
                        x_t = x_t.permute(0, 3, 1, 2)
                    y_t = torch.tensor(y, dtype=torch.long, device=device)
                    accs.append(eval_fn(x_t, y_t))

                val_acc = float(torch.stack(accs).mean()) if accs else 0.0
                model.train()

                wandb.log({
                    "val/val_acc": val_acc,
                    "best_val_acc": max(best_val_acc, val_acc),
                    "training_time_s": time.time() - t0,
                }, step=step)

                pbar.set_postfix(val=f"{val_acc:.4f}", best=f"{best_val_acc:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc, best_step = val_acc, step
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= patience:
                    print(f"Early stop at step {step}")
                    break

            # ── Train step ───────────────────────
            x, y = next(train_iter)
            x_t = torch.tensor(x, dtype=torch.float32, device=device)
            if x_t.shape[-1] in [1, 3]:
                x_t = x_t.permute(0, 3, 1, 2)
            y_t = torch.tensor(y, dtype=torch.long, device=device)

            loss = step_fn(x_t, y_t)
            wandb.log({"train/loss": float(loss)}, step=step)
            step += 1
            pbar.update(1)

    # ── Final evaluation ─────────────────────────
    total_time = time.time() - t0
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    test_accs = []
    for x, y in test_loader:
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        if x_t.shape[-1] in [1, 3]:
            x_t = x_t.permute(0, 3, 1, 2)
        y_t = torch.tensor(y, dtype=torch.long, device=device)
        test_accs.append(eval_fn(x_t, y_t))

    test_acc = float(torch.stack(test_accs).mean()) if test_accs else 0.0

    wandb.log({
        "test/test_acc": test_acc,
        "best_val_acc": best_val_acc,
        "total_training_time_s": total_time,
    }, step=step)

    print(f"Test Acc: {test_acc:.4f}  Best Val: {best_val_acc:.4f}  Time: {total_time:.1f}s")
    wandb.finish()

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    train()
