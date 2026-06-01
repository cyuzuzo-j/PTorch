##################################################
###   Benchmark — ptorch (Projection baseline) ###
###   ViT with Projection Optimizers          ###
##################################################
import sys, os, argparse, gc, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import tqdm
from ptorch.core.ops import process_activation_target
from frameworks.ptorch.nn.modules import Linear, CrossEntropy
from frameworks.ptorch.core.ops import SoftmaxProjection
from frameworks.ptorch.nn.modules_experimental import RMSNorm, MultiheadAttention, SeqMaxPool, Branch, SeqAvgPool
from frameworks.ptorch import config as ptorch_config
import frameworks.ptorch.optim_static as ptorch_optim_static
from frameworks.ptorch.core.overrides import apply_overrides

from experiments.shared.data import MNISTDataModule, InfiniteCifarDataModule

ptorch_config.config.use_projections = True

FRAMEWORK = "ptorch"
DATASETS = {"MNIST": MNISTDataModule, "CIFAR10": InfiniteCifarDataModule}
OPTIM_MODULES = vars(ptorch_optim_static)
# ── Residual addition with independent target propagation ───────────────────

class ResidualAdd(torch.autograd.Function):
    """
    Addition for residual connections where the two inputs are independent.
    Properly projects the target onto the addition constraint by splitting
    the residual difference equally between the two branches.
    """
    @staticmethod
    def forward(ctx, skip, out):
        z = skip + out
        ctx.save_for_backward(skip, out, z)
        return z

    @staticmethod
    def backward(ctx, z_target):
        skip, out, z = ctx.saved_tensors
        
        # Calculate the residual displacement
        delta = (z_target - z) / 2.0
        
        # Project targets for both branches
        skip_target = skip + delta
        out_target = out + delta
        
        return skip_target, out_target

# ── Model ────────────────────────────────────────
class MNISTAttention_PTorch(nn.Module):
    def __init__(self, patch_size=7, emb_dim=64, num_heads=4, norm='l2'):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = (28 // patch_size) ** 2
        self.patch_dim = patch_size * patch_size
        self.branch_res = Branch(2)

        self.embedding = Linear(self.patch_dim, emb_dim, norm=norm)
        self.norm1 = RMSNorm()
        self.branch_qkv = Branch(3)
        self.attn = MultiheadAttention(emb_dim, num_heads, num_heads, 10000.0, 1.0, norm=norm)

        self.mlp_head = Linear(emb_dim*self.num_patches, 10, norm=norm)

    def forward(self, x):
        B = x.shape[0]
        # ensure shape is B, 1, 28, 28 (MNIST standard in this repo)
        if x.ndim == 3: # (B, 28, 28)
            x = x.unsqueeze(1)
        elif x.ndim == 4 and x.shape[-1] == 1: # (B, 28, 28, 1)
            x = x.permute(0, 3, 1, 2)

        # split into patches
        patches = x.unfold(2, self.patch_size, self.patch_size)\
                   .unfold(3, self.patch_size, self.patch_size)

        patches = patches.contiguous().view(B, -1, self.patch_dim)

        # Project patches
        tokens = self.embedding(patches)
        tokens_att, tokens_skip = self.branch_res(tokens)
        tokens_norm = self.norm1(tokens_att)
        q, k, v = self.branch_qkv(tokens_norm)

        attn_out = self.attn(q, k, v)
        output = ResidualAdd.apply(tokens_skip, attn_out)
        #pooled = self.pool(attn_out)
        pooled = output.reshape(-1,self.num_patches*64 )
        return self.mlp_head(pooled)

# ── Training ─────────────────────────────────────
def run(cfg, task_cfg, batch_size, run_number, device,
        norm='l2', opt_name=None, opt_kwargs=None, loss_name='CrossEntropy'):
    seed = cfg["random_seed"]
    run_seed = seed + run_number
    torch.manual_seed(run_seed)

    dataset_name = task_cfg.get("name")
    dataset_cls = DATASETS[dataset_name]
    ds = dataset_cls(batch_size=batch_size, seed=run_seed)
    train_iter = ds.train_iterator()
    val_loader  = ds.val_dataloader()
    test_loader = ds.test_dataloader()

    model = MNISTAttention_PTorch(
        patch_size=task_cfg.get("patch_size", 7),
        emb_dim=task_cfg.get("emb_dim", 64),
        num_heads=task_cfg.get("num_heads", 4),
        norm=norm
    ).to(device)
    
    opt_name   = opt_name or cfg.get("ptorch_optimizer", "ProjectionMuon")
    opt_kwargs = opt_kwargs if opt_kwargs is not None else cfg.get("ptorch_optimizer_kwargs", {"lr": 0.01})
    optimizer  = OPTIM_MODULES[opt_name](model.parameters(), **opt_kwargs)
    
    criterion = getattr(nn, loss_name, None)
    if criterion is None:
        # try ptorch modules
        import frameworks.ptorch.nn.modules as pnn
        criterion = getattr(pnn, loss_name, CrossEntropy)
    criterion = criterion()

    tag = f"{FRAMEWORK}_norm={norm}_opt={opt_name}_loss={loss_name}"
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
                if xv.ndim == 4 and xv.shape[-1] == 1:
                    xv = xv.permute(0, 3, 1, 2)
                yv = torch.tensor(yv, dtype=torch.long,  device=device)
                accs.append((model(xv).argmax(dim=-1) == yv).float().mean())
        model.train()
        return float(torch.stack(accs).mean())

    best_val_acc, best_state, best_step = 0.0, None, 0
    last_val_acc = 0.0
    no_improve = 0
    t0 = time.time()
    step = 0

    with tqdm.tqdm(total=cfg["max_steps"], unit="step", desc=tag) as pbar:
        while step < cfg["max_steps"]:
            if step % cfg["eval_every"] == 0:
                val_acc = eval_acc(val_loader)
                last_val_acc = val_acc
                elapsed = time.time() - t0
                csv_rows.append({
                    "framework": FRAMEWORK, "task": dataset_name,
                    "norm": norm, "optimizer": opt_name, "loss": loss_name,
                    "run": run_number, "step": step,
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
            if x_batch.ndim == 4 and x_batch.shape[-1] == 1:
                x_batch = x_batch.permute(0, 3, 1, 2)
            y_batch = torch.tensor(y_np, dtype=torch.long,  device=device)
            y_oh = F.one_hot(y_batch, num_classes=10).float()

            optimizer.zero_grad()
            preds = model(x_batch)
            loss_val = criterion(preds, y_oh)
            if isinstance(loss_val, torch.Tensor):
                loss_val.sum().backward()
            optimizer.step()

            train_acc = float((preds.argmax(dim=-1) == y_batch).float().mean())

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
    
    csv_rows.append({
        "framework": FRAMEWORK, "task": dataset_name,
        "norm": norm, "optimizer": opt_name, "loss": loss_name,
        "run": run_number, "step": step,
        "val_acc": test_acc, "wall_time_s": total_time,
    })

    df = pd.DataFrame(csv_rows)
    df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)
    return test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PTorch Attention benchmark")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    norms = cfg.get("norms", ["l2"])
    losses = cfg.get("losses", ["CrossEntropy"])
    optimizers = cfg.get("optimizers", [{"name": "ProjectionMuon", "kwargs": {"lr": 0.01}}])

    for batch_size in cfg["batch_sizes"]:
        for task_cfg in cfg["tasks"]:
            for norm in norms:
                for opt_entry in optimizers:
                    for loss_name in losses:
                        opt_name = opt_entry["name"]
                        opt_kwargs = opt_entry.get("kwargs", {})
                        print(f"\n{'='*60}\n{FRAMEWORK} | {task_cfg['name']} | norm={norm} | opt={opt_name} | loss={loss_name}")
                        for run_number in range(1, cfg["num_runs"] + 1):
                            run(cfg, task_cfg, batch_size, run_number, device,
                                norm=norm, opt_name=opt_name, opt_kwargs=opt_kwargs, loss_name=loss_name)
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
