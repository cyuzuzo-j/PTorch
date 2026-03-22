##################################################
###   Benchmark — Pure PyTorch Baseline       ###
###   Standard PyTorch Conv + MLP Head       ###
##################################################
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))

import gc
import yaml
import torch
import torch.nn as tnn
import torch.nn.functional as F

from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, ReLU

from experiments.shared.data import CIFAR10DataModule
import tqdm, time
import wandb

FRAMEWORK = "torch_baseline"
CFG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

DATASETS = {"CIFAR10": CIFAR10DataModule}

# ── Model ────────────────────────────────────────
class StandardModel(tnn.Module):
    def __init__(self, hidden, channels, classes, image_size):
        super().__init__()
        # 1. Standard PyTorch Convolutional Feature Extractor
        self.conv_features = tnn.Sequential(
            Conv2d(channels, 32, kernel_size=3, padding=1),
            tnn.ReLU(inplace=True),
            MaxPool2d(2, 2),
            Conv2d(32, 64, kernel_size=3, padding=1),
            tnn.ReLU(inplace=True),
            MaxPool2d(2, 2),
            Flatten()
        )
        
        # Calculate flattened feature size (e.g., for 32x32 image with 2 max pools: 64 * 8 * 8)
        feature_map_size = image_size // 4
        in_features = 64 * feature_map_size * feature_map_size
        
        # 3. Standard PyTorch MLP Head
        last = in_features
        self.mlp_head = tnn.ModuleList()
        for f in hidden:
            self.mlp_head.append(Linear(last, f))
            self.mlp_head.append(ReLU(inplace=True))
            last = f
        self.out = Linear(last, classes)

    def forward(self, x):
        features = self.conv_features(x)
        
        out = features
        for i in range(0, len(self.mlp_head), 2):
            out = self.mlp_head[i](out)      # Linear
            out = self.mlp_head[i + 1](out)  # ReLU
        out = self.out(out)
        
        return out


# ── Training ─────────────────────────────────────
def run(cfg, task_cfg, batch_size, run_number, device):
    seed = cfg["random_seed"]
    torch.manual_seed(seed + run_number)

    dataset_cls = DATASETS[task_cfg["name"]]
    ds = dataset_cls(batch_size=batch_size, seed=seed)
    train_iter = ds.train_iterator()
    val_loader  = ds.val_dataloader()
    test_loader = ds.test_dataloader()

    model = StandardModel(task_cfg["hidden"], task_cfg["channels"], task_cfg["classes"], task_cfg["image_size"]).to(device)
    
    # Optimizer for the standard PyTorch model
    torch_opt_name = cfg["torch_optimizer"]
    torch_opt_kwargs = cfg.get("torch_optimizer_kwargs", {})
    torch_optimizer = getattr(torch.optim, torch_opt_name)(model.parameters(), **torch_opt_kwargs)
    
    run_name = f"{cfg.get('experiment_name', 'hybrid')}_{FRAMEWORK}_{task_cfg['name']}_bs{batch_size}_run{run_number}"
    run = wandb.init(project="pjax", name=run_name, group=cfg.get('experiment_name', 'hybrid'))
    
    wandb.config.update({
        "framework": FRAMEWORK,
        "task": task_cfg["name"],
        "hidden": task_cfg["hidden"],
        "batch_size": batch_size,
        "seed": seed,
        "run_number": run_number,
        "max_steps": cfg["max_steps"],
        "torch_optimizer": torch_opt_name,
        **{f"torch_opt_{k}": v for k, v in torch_opt_kwargs.items()},
    })

    def step_fn(x, y):
        # Forward pass
        logits = model(x)
        
        torch_optimizer.zero_grad()
        loss = F.cross_entropy(logits, y.long())
        loss.backward()
        torch_optimizer.step()
        
        return float(loss.detach())

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
                    torch.tensor(x, dtype=torch.float32, device=device).permute(0, 3, 1, 2), # NCHW
                    torch.tensor(y, dtype=torch.long,  device=device))
                    for x, y in val_loader]
                val_acc = float(torch.stack(accs).mean() if len(accs) > 0 else 0)
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
            # Dataloader outputs NHWC, PyTorch Conv2d wants NCHW
            x = torch.tensor(x, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
            y = torch.tensor(y, dtype=torch.long, device=device)
            loss = step_fn(x, y)
            
            wandb.log({"train/loss": loss}, step=step)
            step += 1
            pbar.update(1)
            if cfg["max_steps"] and step >= cfg["max_steps"]:
                break

    total_time = time.time() - t0
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    test_accs = [eval_fn(
        torch.tensor(x, dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        torch.tensor(y, dtype=torch.long,  device=device))
        for x, y in test_loader]
    final_acc = float(torch.stack(test_accs).mean() if len(test_accs) > 0 else 0)
    print(f"Test Acc: {final_acc:.4f}  Time: {total_time:.1f}s")
    wandb.log({"test/test_acc": final_acc}, step=step)
    wandb.log({"total_training_time_s": total_time}, step=step)
    wandb.finish()
    return final_acc, best_val_acc, best_step, total_time


if __name__ == "__main__":
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.disable()
    
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
