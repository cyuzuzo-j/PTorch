##################################################
###   Benchmark — ptorch (Projection baseline) ###
###   Regular ViT with Projection Optimizers   ###
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
from frameworks.ptorch.nn.modules import Linear, CrossEntropy, LeakyReLU, ReLU
from frameworks.ptorch.nn.modules_experimental import RMSNorm, MultiheadAttention, Branch, SeqAvgPool, SeqMaxPool
from frameworks.ptorch import config as ptorch_config
import frameworks.ptorch.optim_static as ptorch_optim_static

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

        # Gain-1 residual copy (backprop-residual analog). The minimum-norm
        # projection onto {skip+out=z_target} splits the displacement equally
        # (delta/2 each); compounded over depth that attenuates the teaching
        # signal ~2x per add (and ~4x per block once the Branch averages the
        # skip target against the near-dead attn/mlp branch), freezing the body.
        # Like the +1 identity gradient that lets ResNets train deep, route the
        # FULL displacement to both addends so the skip path carries the signal
        # down the backbone with gain 1 and the sublayers keep full signal.
        # This leaves the strict min-norm constraint plane (sum overshoots by
        # delta), recovered by the alternating projections over steps.
        delta = z_target - z

        skip_target = skip + delta
        out_target = out + delta

        return skip_target, out_target

# ── ViT Components ────────────────────────────────────────

class MLP(nn.Module):
    """ Standard MLP block used in Transformer encoders. """
    def __init__(self, in_features, hidden_features, norm='l2'):
        super().__init__()
        self.fc1 = Linear(in_features, hidden_features, norm=norm)
        # ReLU has projection variants for both 'l2' and 'linf', LeakyReLU only has 'l2',
        # which makes the linf-Linear/L2-act/linf-Linear chain geometrically inconsistent.
        self.act = ReLU(norm=norm) if norm == 'linf' else LeakyReLU()
        self.fc2 = Linear(hidden_features, in_features, norm=norm)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class TransformerEncoderBlock(nn.Module):
    """ Standard Pre-Norm Transformer Encoder Block. """
    def __init__(self, emb_dim, num_heads, mlp_ratio=4.0, norm='l2'):
        super().__init__()

        # Attention branch
        self.norm1 = RMSNorm()
        self.branch_res_attn = Branch(2)
        self.branch_qkv = Branch(3)
        self.attn = MultiheadAttention(emb_dim, num_heads, num_heads, 10000.0, 1.0, norm=norm)

        # MLP branch
        self.norm2 = RMSNorm()
        self.branch_res_mlp = Branch(2)
        mlp_hidden_dim = int(emb_dim * mlp_ratio)
        self.mlp = MLP(emb_dim, mlp_hidden_dim, norm=norm)

    def forward(self, x):
        # Attention Sublayer
        x_res1, x_skip1 = self.branch_res_attn(x)
        x_norm1 = self.norm1(x_res1)
        q, k, v = self.branch_qkv(x_res1)
        attn_out = self.attn(q, k, v)
        x = ResidualAdd.apply(x_skip1, attn_out)

        # MLP Sublayer
        x_res2, x_skip2 = self.branch_res_mlp(x)
        x_norm2 = self.norm2(x_res2)
        mlp_out = self.mlp(x_res2)
        x = ResidualAdd.apply(x_skip2, mlp_out)

        return x

class ViT_PTorch(nn.Module):
    """ Standard Vision Transformer Architecture. """
    def __init__(self, img_size=28, patch_size=7, in_chans=1, num_classes=10,
                 emb_dim=64, depth=4, num_heads=4, mlp_ratio=4.0, norm='l2',
                 head_pool='flatten'):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_chans * patch_size * patch_size
        self.head_pool = head_pool

        # Patch Embedding
        self.patch_embed = Linear(self.patch_dim, emb_dim, norm=norm)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(emb_dim, num_heads, mlp_ratio, norm)
            for _ in range(depth)
        ])

        if head_pool == 'flatten':
            # No norm before flatten+head: RMSNormProjection.backward reconstructs
            # x_bar = sigma_bar * z_bar with z_bar renormalized to magnitude sqrt(D),
            # which wipes per-token magnitude info that the flattened head needs.
            self.head = Linear(emb_dim * self.num_patches, num_classes, norm=norm)
        else:
            # Narrow readout: pool tokens (B,T,D)->(B,D) then a D->num_classes head.
            self.pool = SeqAvgPool() if head_pool == 'avg' else SeqMaxPool()
            self.head = Linear(emb_dim, num_classes, norm=norm)

    def forward(self, x):
        B = x.shape[0]
        # ensure shape is B, 1, 28, 28
        if x.ndim == 3: # (B, 28, 28)
            x = x.unsqueeze(1)
        elif x.ndim == 4 and x.shape[-1] == 1: # (B, 28, 28, 1)
            x = x.permute(0, 3, 1, 2)

        # 1. Patchify: shape (B, num_patches, patch_dim)
        # unfold -> (B, C, nph, npw, ph, pw); permute so each token groups all
        # channels of one spatial patch (C=1 makes the permute a no-op for MNIST).
        patches = x.unfold(2, self.patch_size, self.patch_size)\
                   .unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(B, -1, self.patch_dim)

        # 2. Linear Projection
        tokens = self.patch_embed(patches)

        # 3. Pass through Transformer Blocks (Positional encoding handled by RoPE inside MultiheadAttention)
        for block in self.blocks:
            tokens = block(tokens)

        # 4. Classification Head. Return raw logits — CrossEntropyProjection's
        # proximal iteration `x += lmbda * (target - softmax(x))` is in logit space.
        if self.head_pool == 'flatten':
            logits = self.head(tokens.reshape(B, -1))
        else:
            logits = self.head(self.pool(tokens))
        return logits

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

    model = ViT_PTorch(
        img_size=28 if dataset_name == "MNIST" else 32, # handle CIFAR
        in_chans=1 if dataset_name == "MNIST" else 3,
        patch_size=task_cfg.get("patch_size", 7 if dataset_name=="MNIST" else 8),
        emb_dim=task_cfg.get("emb_dim", 64),
        depth=task_cfg.get("depth", 4),
        num_heads=task_cfg.get("num_heads", 4),
        mlp_ratio=task_cfg.get("mlp_ratio", 4.0),
        norm=norm,
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
                if xv.ndim == 4 and xv.shape[-1] in [1, 3]: # Check for channel last
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
            y_batch = torch.tensor(y_np, dtype=torch.long,  device=device)
            y_oh = F.one_hot(y_batch, num_classes=10).float()

            optimizer.zero_grad()
            preds = model(x_batch)
            loss_val = criterion(preds, y_oh)
            loss_val.sum().backward()
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
    parser = argparse.ArgumentParser(description="PTorch ViT benchmark")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    parser.add_argument("--max-steps", type=int, default=None, help="Override cfg['max_steps']")
    parser.add_argument("--num-runs", type=int, default=None, help="Override cfg['num_runs']")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.max_steps is not None: cfg["max_steps"] = args.max_steps
    if args.num_runs is not None: cfg["num_runs"] = args.num_runs
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
