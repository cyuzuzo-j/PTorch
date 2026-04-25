##################################################
###   Benchmark — ptorch (Projection baseline) ###
###   ViT with Projection Optimizers          ###
##################################################
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))

import gc
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from experiments.shared.data import MNISTDataModule, InfiniteCifarDataModule
from frameworks.ptorch.nn.modules import Linear, Mean
from frameworks.ptorch.nn.experimental_modules import MultiHeadAttention
from frameworks.ptorch.core.ops import CrossEntropyProjection
from frameworks.ptorch.core.overrides import apply_overrides
import ptorch.optim_static as optim_static
import tqdm, time
import wandb

FRAMEWORK = "ptorch"
CFG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

DATASETS = {"MNIST": MNISTDataModule, "CIFAR10": InfiniteCifarDataModule}

apply_overrides()

# ── Model ────────────────────────────────────────
class MNISTAttention_PTorch(nn.Module):
    def __init__(self, patch_size=7, emb_dim=64, num_heads=4):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = (28 // patch_size) ** 2
        self.patch_dim = patch_size * patch_size

        # Ptorch custom MultiHeadAttention which uses SimplexProjection internally
        self.attn = MultiHeadAttention(self.patch_dim, self.patch_dim // num_heads, num_heads, use_rotary=True)
        self.mean = Mean(1)
        self.mlp_head = Linear(self.patch_dim, 10)

    def forward(self, x):
        B = x.shape[0]
        # ensure shape is B, 1, 28, 28
        x = x.reshape(B, 1, 28, 28)

        # split into patches
        patches = x.unfold(2, self.patch_size, self.patch_size)\
                   .unfold(3, self.patch_size, self.patch_size)

        patches = patches.contiguous().view(B, -1, self.patch_dim)

        # Project patches
        #tokens = self.embedding(patches)

        # ptorch MultiHeadAttention only returns the output, no weights
        attn_out = self.attn(patches)
        
        x_attn =  attn_out
        
        # Mean pooling across patches
        pooled_out = self.mean(x_attn)

        return self.mlp_head(pooled_out)

# ── Training ─────────────────────────────────────
def run(cfg, task_cfg, batch_size, run_number, device):
    seed = cfg["random_seed"]
    torch.manual_seed(seed + run_number)

    dataset_cls = DATASETS[task_cfg["name"]]
    ds = dataset_cls(batch_size=batch_size, seed=seed)
    train_iter = ds.train_iterator()
    val_loader  = ds.val_dataloader()
    test_loader = ds.test_dataloader()

    model = MNISTAttention_PTorch(
        patch_size=task_cfg.get("patch_size", 7),
        emb_dim=task_cfg.get("emb_dim", 64),
        num_heads=task_cfg.get("num_heads", 4)
    ).to(device)
    
    opt_name   = cfg["ptorch_optimizer"] if "ptorch_optimizer" in cfg else cfg["torch_optimizer"]
    opt_kwargs = cfg.get("ptorch_optimizer_kwargs", {}) if "ptorch_optimizer" in cfg else cfg.get("torch_optimizer_kwargs", {})
    
    if hasattr(optim_static, opt_name):
        optimizer = getattr(optim_static, opt_name)(model.parameters(), **opt_kwargs)
    else:
        optimizer = getattr(torch.optim, opt_name)(model.parameters(), **opt_kwargs)

    run = wandb.init(project="pjax", name=cfg["experiment_name"])
    wandb.config.update({
        "framework": FRAMEWORK,
        "task": task_cfg["name"],
        "patch_size": task_cfg.get("patch_size", 7),
        "emb_dim": task_cfg.get("emb_dim", 64),
        "num_heads": task_cfg.get("num_heads", 4),
        "optimizer": opt_name,
        **{f"opt_{k}": v for k, v in opt_kwargs.items()},
        "batch_size": batch_size,
        "seed": seed,
        "run_number": run_number,
        "max_steps": cfg["max_steps"],
        "eval_every": cfg["eval_every"],
        "patience": cfg["patience"],
    })

    def step_fn(x, y):
        optimizer.zero_grad()
        targets = F.one_hot(y, num_classes=10).float()
        
        preds = model(x)
        projected = CrossEntropyProjection.apply(preds, targets)
        
        projected.sum().backward()
        optimizer.step()
        
        with torch.no_grad():
            loss = F.cross_entropy(preds, y.long())
        return loss

    def eval_fn(x, y):
        with torch.no_grad():
            return (model(x).argmax(dim=-1) == y).float().mean()

    best_val_acc, best_state, best_step = 0.0, None, 0
    no_improve = 0
    step = 0
    t0 = time.time()

    with tqdm.tqdm(unit="step") as pbar:
        while True:
            if step % cfg["eval_every"] == 0:
                model.eval()
                accs = [eval_fn(
                    torch.tensor(x, dtype=torch.float32, device=device),
                    torch.tensor(y, dtype=torch.long,  device=device))
                    for x, y in val_loader]
                val_acc = float(torch.stack(accs).mean())
                model.train()
                wandb.log({"val/val_acc": val_acc}, step=step)
                wandb.log({"training_time_s": time.time() - t0}, step=step)
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

            x, y = next(train_iter)
            loss = step_fn(
                torch.tensor(x, dtype=torch.float32, device=device),
                torch.tensor(y, dtype=torch.long,  device=device))
            wandb.log({"train/loss": float(loss)}, step=step)
            step += 1
            pbar.update(1)
            if cfg["max_steps"] and step >= cfg["max_steps"]:
                break

    total_time = time.time() - t0
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    test_accs = [eval_fn(
        torch.tensor(x, dtype=torch.float32, device=device),
        torch.tensor(y, dtype=torch.long,  device=device))
        for x, y in test_loader]
    final_acc = float(torch.stack(test_accs).mean())
    print(f"Test Acc: {final_acc:.4f}  Time: {total_time:.1f}s")
    wandb.log({"test/test_acc": final_acc}, step=step)
    wandb.log({"total_training_time_s": total_time}, step=step)
    wandb.finish()
    return final_acc, best_val_acc, best_step, total_time


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
