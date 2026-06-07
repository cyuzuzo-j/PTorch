##################################################
###   Benchmark — torch (PyTorch baseline)      ###
###   Standard ViT with stock optimizers        ###
##################################################
import sys, os, argparse, gc, time
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import tqdm

from experiments.shared.data import MNISTDataModule, InfiniteCifarDataModule

FRAMEWORK = "torch"
DATASETS = {"MNIST": MNISTDataModule, "CIFAR10": InfiniteCifarDataModule}


# ── ViT Components ────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TransformerEncoderBlock(nn.Module):
    """Standard Pre-Norm Transformer encoder block."""
    def __init__(self, emb_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = MLP(emb_dim, int(emb_dim * mlp_ratio))

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ViT_Torch(nn.Module):
    """Standard Vision Transformer with learnable positional embeddings."""
    def __init__(self, img_size=28, patch_size=7, in_chans=1, num_classes=10,
                 emb_dim=64, depth=4, num_heads=4, mlp_ratio=4.0,
                 head_pool='flatten'):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_chans * patch_size * patch_size
        self.head_pool = head_pool

        self.patch_embed = nn.Linear(self.patch_dim, emb_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, emb_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(emb_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_dim)

        if head_pool == 'flatten':
            self.head = nn.Linear(emb_dim * self.num_patches, num_classes)
        else:
            self.head = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        if x.ndim == 3:
            x = x.unsqueeze(1)
        elif x.ndim == 4 and x.shape[-1] == 1:
            x = x.permute(0, 3, 1, 2)

        patches = x.unfold(2, self.patch_size, self.patch_size)\
                   .unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(B, -1, self.patch_dim)

        tokens = self.patch_embed(patches) + self.pos_embed

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm(tokens)

        if self.head_pool == 'flatten':
            logits = self.head(tokens.reshape(B, -1))
        elif self.head_pool == 'avg':
            logits = self.head(tokens.mean(dim=1))
        else:
            logits = self.head(tokens.max(dim=1).values)
        return logits


# ── Training ─────────────────────────────────────
def run(cfg, task_cfg, batch_size, run_number, device,
        opt_name=None, opt_kwargs=None, loss_name='CrossEntropy'):
    seed = cfg["random_seed"]
    run_seed = seed + run_number
    torch.manual_seed(run_seed)

    dataset_name = task_cfg.get("name")
    dataset_cls = DATASETS[dataset_name]
    ds = dataset_cls(batch_size=batch_size, seed=run_seed)
    train_iter = ds.train_iterator()
    val_loader = ds.val_dataloader()
    test_loader = ds.test_dataloader()

    model = ViT_Torch(
        img_size=28 if dataset_name == "MNIST" else 32,
        in_chans=1 if dataset_name == "MNIST" else 3,
        patch_size=task_cfg.get("patch_size", 7 if dataset_name == "MNIST" else 8),
        emb_dim=task_cfg.get("emb_dim", 64),
        depth=task_cfg.get("depth", 4),
        num_heads=task_cfg.get("num_heads", 4),
        mlp_ratio=task_cfg.get("mlp_ratio", 4.0),
    ).to(device)

    opt_name = opt_name or cfg.get("torch_optimizer", "AdamW")
    opt_kwargs = opt_kwargs if opt_kwargs is not None else cfg.get("torch_optimizer_kwargs", {"lr": 1e-3})
    optimizer = getattr(torch.optim, opt_name)(model.parameters(), **opt_kwargs)

    # Mirror the ptorch tag layout so plot_results.py can join both CSVs.
    norm = "l2"
    tag = f"{FRAMEWORK}_ViT_norm={norm}_opt={opt_name}_loss={loss_name}"
    results_dir = os.path.join(os.path.dirname(__file__), cfg.get("results_dir", "results"))
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"{tag}_{dataset_name}.csv")
    csv_rows = []

    def eval_acc(loader):
        model.eval()
        accs = []
        with torch.no_grad():
            for xv, yv in loader:
                xv = torch.tensor(xv, dtype=torch.float32, device=device)
                if xv.ndim == 4 and xv.shape[-1] in [1, 3]:
                    xv = xv.permute(0, 3, 1, 2)
                yv = torch.tensor(yv, dtype=torch.long, device=device)
                accs.append((model(xv).argmax(dim=-1) == yv).float().mean())
        model.train()
        return float(torch.stack(accs).mean())

    best_val_acc, best_state, best_step = 0.0, None, 0
    last_val_acc = 0.0
    no_improve = 0
    t0 = time.time()
    step = 0
    train_acc_sum, train_acc_count = 0.0, 0
    last_train_acc = 0.0

    with tqdm.tqdm(total=cfg["max_steps"], unit="step", desc=tag) as pbar:
        while step < cfg["max_steps"]:
            if step % cfg["eval_every"] == 0:
                val_acc = eval_acc(val_loader)
                last_val_acc = val_acc
                avg_train_acc = (train_acc_sum / train_acc_count) if train_acc_count > 0 else 0.0
                last_train_acc = avg_train_acc
                train_acc_sum, train_acc_count = 0.0, 0
                elapsed = time.time() - t0
                csv_rows.append({
                    "framework": FRAMEWORK, "task": dataset_name,
                    "norm": norm, "optimizer": opt_name, "loss": loss_name,
                    "run": run_number, "step": step,
                    "train_acc": avg_train_acc,
                    "val_acc": val_acc, "wall_time_s": elapsed,
                })
                if val_acc > best_val_acc:
                    best_val_acc, best_step = val_acc, step
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= cfg["patience"]:
                    print(f"Early stop at step {step}")
                    break

            x_np, y_np = next(train_iter)
            x_batch = torch.tensor(x_np, dtype=torch.float32, device=device)
            if x_batch.ndim == 4 and x_batch.shape[-1] in [1, 3]:
                x_batch = x_batch.permute(0, 3, 1, 2)
            y_batch = torch.tensor(y_np, dtype=torch.long, device=device)

            optimizer.zero_grad()
            preds = model(x_batch)
            loss_val = F.cross_entropy(preds, y_batch)
            loss_val.backward()
            optimizer.step()

            train_acc = float((preds.argmax(dim=-1) == y_batch).float().mean())
            train_acc_sum += train_acc
            train_acc_count += 1

            step += 1
            pbar.set_postfix(
                train_acc=f"{train_acc:.4f}",
                val_acc=f"{last_val_acc:.4f}",
                best=f"{best_val_acc:.4f}",
            )
            pbar.update(1)

    total_time = time.time() - t0
    if best_state:
        model.load_state_dict(best_state)
    test_acc = eval_acc(test_loader)
    print(f"Test Acc: {test_acc:.4f}  Time: {total_time:.1f}s")

    final_train_acc = (train_acc_sum / train_acc_count) if train_acc_count > 0 else last_train_acc
    csv_rows.append({
        "framework": FRAMEWORK, "task": dataset_name,
        "norm": norm, "optimizer": opt_name, "loss": loss_name,
        "run": run_number, "step": step,
        "train_acc": final_train_acc,
        "val_acc": test_acc, "wall_time_s": total_time,
    })

    df = pd.DataFrame(csv_rows)
    df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)
    return test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Torch ViT baseline benchmark")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    parser.add_argument("--max-steps", type=int, default=None, help="Override cfg['max_steps']")
    parser.add_argument("--num-runs", type=int, default=None, help="Override cfg['num_runs']")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.max_steps is not None: cfg["max_steps"] = args.max_steps
    if args.num_runs is not None: cfg["num_runs"] = args.num_runs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    losses = cfg.get("losses", ["CrossEntropy"])
    optimizers = cfg.get("torch_optimizers", [
        {"name": cfg.get("torch_optimizer", "AdamW"),
         "kwargs": cfg.get("torch_optimizer_kwargs", {"lr": 1e-3, "weight_decay": 0.01})}
    ])

    for batch_size in cfg["batch_sizes"]:
        for task_cfg in cfg["tasks"]:
            for opt_entry in optimizers:
                for loss_name in losses:
                    opt_name = opt_entry["name"]
                    opt_kwargs = opt_entry.get("kwargs", {})
                    print(f"\n{'='*60}\n{FRAMEWORK} | {task_cfg['name']} | opt={opt_name} | loss={loss_name}")
                    for run_number in range(1, cfg["num_runs"] + 1):
                        run(cfg, task_cfg, batch_size, run_number, device,
                            opt_name=opt_name, opt_kwargs=opt_kwargs, loss_name=loss_name)
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
