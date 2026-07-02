##################################################
###   Benchmark — ptorch (Projection baseline) ###
###   Regular ViT with Projection Optimizers   ###
##################################################
import sys, os, argparse, gc, math, time
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import tqdm
from ptorch.nn.modules import Linear, CrossEntropy, LeakyReLU, ReLU
from ptorch.nn.modules_experimental import RMSNorm, DyT, MultiheadAttention, Branch, SeqAvgPool, SeqMaxPool
from ptorch import config as ptorch_config
import ptorch.optim_static as ptorch_optim_static

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
    def forward(ctx, skip, out, mode="full"):
        z = skip + out
        ctx.save_for_backward(skip, out, z)
        ctx.mode = mode
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
        # mode="half" restores the strict min-norm split (delta/2 each) as a
        # fallback when full-delta routing + delta_sum branches overshoot.
        delta = z_target - z
        if ctx.mode == "half":
            delta = delta / 2.0

        skip_target = skip + delta
        out_target = out + delta

        return skip_target, out_target, None

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
    """ Transformer Encoder Block with optional pre-norm.

    prenorm:
      "none": no pre-norm (historical effective behavior — earlier revisions
              computed norm1/norm2 but never consumed their outputs).
      "rms":  standard pre-norm wiring with the RMSNorm projection.
      "dyt":  normalization-free DyT (hardtanh) layer in the pre-norm slot.
    """
    def __init__(self, emb_dim, num_heads, mlp_ratio=4.0, norm='l2',
                 prenorm='none', qk_match_g_to_omega=False, residual_mode='full'):
        super().__init__()
        self.residual_mode = residual_mode

        if prenorm == 'rms':
            self.norm1, self.norm2 = RMSNorm(), RMSNorm()
        elif prenorm == 'dyt':
            self.norm1, self.norm2 = DyT(), DyT()
        elif prenorm == 'none':
            self.norm1 = self.norm2 = None
        else:
            raise ValueError(f"Unknown prenorm mode: {prenorm}")

        # Attention branch
        self.branch_res_attn = Branch(2)
        self.branch_qkv = Branch(3)
        self.attn = MultiheadAttention(emb_dim, num_heads, num_heads, 10000.0, 1.0,
                                       norm=norm,
                                       qk_match_g_to_omega=qk_match_g_to_omega)

        # MLP branch
        self.branch_res_mlp = Branch(2)
        mlp_hidden_dim = int(emb_dim * mlp_ratio)
        self.mlp = MLP(emb_dim, mlp_hidden_dim, norm=norm)

    def forward(self, x):
        # Attention Sublayer
        x_res1, x_skip1 = self.branch_res_attn(x)
        h1 = self.norm1(x_res1) if self.norm1 is not None else x_res1
        q, k, v = self.branch_qkv(h1)
        attn_out = self.attn(q, k, v)
        x = ResidualAdd.apply(x_skip1, attn_out, self.residual_mode)

        # MLP Sublayer
        x_res2, x_skip2 = self.branch_res_mlp(x)
        h2 = self.norm2(x_res2) if self.norm2 is not None else x_res2
        mlp_out = self.mlp(h2)
        x = ResidualAdd.apply(x_skip2, mlp_out, self.residual_mode)

        return x

class ViT_PTorch(nn.Module):
    """ Standard Vision Transformer Architecture. """
    def __init__(self, img_size=28, patch_size=7, in_chans=1, num_classes=10,
                 emb_dim=64, depth=4, num_heads=4, mlp_ratio=4.0, norm='l2',
                 head_pool='flatten', prenorm='none', qk_match_g_to_omega=False,
                 residual_mode='full'):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_chans * patch_size * patch_size
        self.head_pool = head_pool

        # Patch Embedding
        self.patch_embed = Linear(self.patch_dim, emb_dim, norm=norm)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(emb_dim, num_heads, mlp_ratio, norm,
                                    prenorm=prenorm,
                                    qk_match_g_to_omega=qk_match_g_to_omega,
                                    residual_mode=residual_mode)
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

    def forward_features(self, x):
        """Everything before the head; returns exactly what the head consumes
        (flattened B×(T·D) tokens or pooled B×D), so a probe trained on these
        features sees the same representation as the projection-trained head."""
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

        if self.head_pool == 'flatten':
            return tokens.reshape(B, -1)
        return self.pool(tokens)

    def forward(self, x):
        # Classification Head. Return raw logits — CrossEntropyProjection's
        # proximal iteration `x += lmbda * (target - softmax(x))` is in logit space.
        return self.head(self.forward_features(x))

# ── Training ─────────────────────────────────────
FLAG_DEFAULTS = {
    "prenorm": "none",            # none | rms | dyt
    "branch_mode": "mean",        # mean | delta_sum
    "softmax_mode": "legacy",     # legacy | rel_floor | simplex_l2
    "qk_match_g": False,          # match g to omega in the QK^T projection
    "rmsnorm_mode": "legacy",     # legacy | exact
    "residual_mode": "full",      # full | half (delta routing in ResidualAdd)
    "muon_acts": False,           # Muon orthogonalization on activation targets
    "muon_acts_lr": 1.0,
    "muon_acts_eps": 1e-2,        # polar-express norm floor for tiny residuals
    "muon_acts_mode": "fixed",    # fixed | rel_row | rel_frob | raw_rel_row
    "muon_acts_max_dim": 4096,    # skip body-rescue on tensors with last_dim > this
                                  # (default 4096 excludes the 8192-dim flatten head)
    "muon_acts_decay": "none",    # none | cosine | linear — anneal muon_acts_lr
                                  # to 0 over muon_acts_decay_steps. The rescale
                                  # is an early-training bootstrap; past the
                                  # 5k-9k peak it compounds noise (train_acc
                                  # collapse in the 30k rel_row run).
    "muon_acts_decay_steps": 0,   # 0 = decay over max_steps
    "momentum": 0.95,             # ProjectionMuonV2 momentum
    "weight_decay": 0.0,          # ProjectionMuonV2 decoupled weight decay
    "ce_lambda": 1.0,             # CrossEntropyProjection proximal step size
    "ce_num_steps": 5,            # CrossEntropyProjection inner iterations
    "lr_schedule": "constant",    # constant | cosine
    # Weight-vs-activation balance of every bilinear solve. With alpha=1 and
    # wide layers (qa = ||act row||^2 >> qb = ||weight col||^2) the exact
    # projection routes ~all displacement into the weights and the backward
    # activation target dies at the first solve (measured: rel_delta ~1e-8 in
    # the whole body). Larger alpha penalizes weight movement and revives the
    # backward signal: t ~ Delta*alpha/(qa + alpha*qb).
    "proj_alpha": 1.0,
    "proj_g": 1.0,                # target-fidelity weight of every solve
    "head_pool": "flatten",       # flatten | avg | max  (flatten => 8192-dim
                                  # head rows => huge qa at the head solve)
    # Direction pipe (TD trace over depth; see experiments/deep/FINDINGS_td.md).
    # td_lambda=0 disables everything (bitwise-identical backward). Note the
    # ViT is not a chain graph: the scalar floor is updated in autograd's
    # engine order across branches, so the trace semantics are heuristic here.
    "td_lambda": 0.0,
    "td_mode": "norm",            # norm (scalar floor) | vector (negative control)
    "td_deflate": False,          # strip the activation-parallel solve artifact
    "td_eps_lin": 1e-2,           # linear-transport amplitude cap (x ||A||)
}


def flag_tag(flags):
    """Short tag of non-default flags for CSV/file names; 'base' if all default."""
    parts = []
    short = {"prenorm": "pn", "branch_mode": "bm", "softmax_mode": "sm",
             "qk_match_g": "qkg", "rmsnorm_mode": "rn", "residual_mode": "rm",
             "muon_acts": "ma", "muon_acts_lr": "malr", "muon_acts_eps": "mae",
             "muon_acts_mode": "mam", "muon_acts_max_dim": "mamd",
             "muon_acts_decay": "mad", "muon_acts_decay_steps": "mads",
             "momentum": "mom", "weight_decay": "wd",
             "ce_lambda": "cel", "ce_num_steps": "ces",
             "lr_schedule": "lrs",
             "proj_alpha": "pa", "proj_g": "pg", "head_pool": "hp",
             "td_lambda": "td", "td_mode": "tdm", "td_deflate": "tdd",
             "td_eps_lin": "tde"}
    for key, default in FLAG_DEFAULTS.items():
        val = flags.get(key, default)
        if val != default:
            parts.append(f"{short[key]}={val}")
    return "+".join(parts) if parts else "base"


def apply_global_flags(flags):
    ptorch_config.config.update("branch_mode", flags.get("branch_mode", "mean"))
    ptorch_config.config.update("softmax_target_mode", flags.get("softmax_mode", "legacy"))
    ptorch_config.config.update("rmsnorm_backward_mode", flags.get("rmsnorm_mode", "legacy"))
    ptorch_config.config.update("use_muon_activations", bool(flags.get("muon_acts", False)))
    ptorch_config.config.update("muon_activations_lr", float(flags.get("muon_acts_lr", 1.0)))
    ptorch_config.config.update("muon_activations_eps", float(flags.get("muon_acts_eps", 1e-2)))
    ptorch_config.config.update("muon_activations_mode", str(flags.get("muon_acts_mode", "fixed")))
    ptorch_config.config.update("muon_activations_max_dim", int(flags.get("muon_acts_max_dim", 4096)))
    ptorch_config.config.update("cross_entropy_lambda", float(flags.get("ce_lambda", 1.0)))
    ptorch_config.config.update("cross_entropy_num_steps", int(flags.get("ce_num_steps", 5)))
    # NOTE: Linear bakes alpha/g at construction time — apply before building
    # the model (run() and diag_targets.py both do).
    proj_alpha = flags.get("proj_alpha", 1.0)
    if isinstance(proj_alpha, str) and proj_alpha == "auto":
        ptorch_config.config.update("projection_alpha", 1.0)
        ptorch_config.config.update("projection_alpha_auto", True)
    else:
        ptorch_config.config.update("projection_alpha", float(proj_alpha))
        ptorch_config.config.update("projection_alpha_auto", False)
    ptorch_config.config.update("projection_g", float(flags.get("proj_g", 1.0)))
    ptorch_config.config.update("td_lambda", float(flags.get("td_lambda", 0.0)))
    ptorch_config.config.update("td_mode", str(flags.get("td_mode", "norm")))
    ptorch_config.config.update("td_deflate", bool(flags.get("td_deflate", False)))
    ptorch_config.config.update("td_eps_lin", float(flags.get("td_eps_lin", 1e-2)))


def run(cfg, task_cfg, batch_size, run_number, device,
        norm='l2', opt_name=None, opt_kwargs=None, loss_name='CrossEntropy',
        flags=None, save_best=False):
    flags = {**FLAG_DEFAULTS, **(flags or {})}
    apply_global_flags(flags)
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
        prenorm=flags["prenorm"],
        qk_match_g_to_omega=bool(flags["qk_match_g"]),
        residual_mode=flags["residual_mode"],
        head_pool=flags["head_pool"],
    ).to(device)

    opt_name   = opt_name or cfg.get("ptorch_optimizer", "ProjectionMuon")
    opt_kwargs = opt_kwargs if opt_kwargs is not None else cfg.get("ptorch_optimizer_kwargs", {"lr": 0.01})
    if opt_name == "ProjectionMuonV2":
        # Flags (not config.yaml) so each arm gets its own CSV via flag_tag.
        opt_kwargs = {**opt_kwargs,
                      "momentum": float(flags["momentum"]),
                      "weight_decay": float(flags["weight_decay"])}
    optimizer  = OPTIM_MODULES[opt_name](model.parameters(), **opt_kwargs)

    # Optional cosine LR decay (constant lr if lr_schedule is "constant").
    # Cosine helps stabilise the late-training divergence we see with
    # mam=rel_row: the body amplification + momentum compound into noise that
    # destroys learned features past step ~10k; a decay to ~0 by max_steps
    # lets the model settle.
    lr_schedule = flags.get("lr_schedule", "constant")
    if lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["max_steps"]
        )
    else:
        scheduler = None

    criterion = getattr(nn, loss_name, None)
    if criterion is None:
        # try ptorch modules
        import ptorch.nn.modules as pnn
        criterion = getattr(pnn, loss_name, CrossEntropy)
    criterion = criterion()

    tag = f"{FRAMEWORK}_ViT_norm={norm}_opt={opt_name}_loss={loss_name}_flags={flag_tag(flags)}"
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
                    "flags": flag_tag(flags),
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

            # Anneal the activation-rescale lr: the amplifier is only needed
            # early, while natural body signal is below the noise floor; left
            # constant it compounds direction-noise past the ~9k peak.
            # Safe per step: ops.py passes the lr into the compiled impl as a
            # tensor, so this never triggers a dynamo recompile.
            if flags["muon_acts"] and flags["muon_acts_decay"] != "none":
                T = int(flags["muon_acts_decay_steps"]) or cfg["max_steps"]
                if flags["muon_acts_decay"] == "cosine":
                    f_decay = 0.5 * (1.0 + math.cos(math.pi * min(step, T) / T))
                else:  # linear
                    f_decay = max(0.0, 1.0 - step / T)
                ptorch_config.config.update(
                    "muon_activations_lr", float(flags["muon_acts_lr"]) * f_decay)

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
            if scheduler is not None:
                scheduler.step()

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

    if save_best and best_state is not None:
        ckpt_path = os.path.join(results_dir, f"ckpt_{tag}_{dataset_name}_run{run_number}.pt")
        torch.save({"state_dict": best_state, "flags": flags,
                    "best_step": best_step, "best_val_acc": best_val_acc,
                    "task_cfg": task_cfg, "norm": norm}, ckpt_path)
        print(f"Saved best state (step {best_step}, val {best_val_acc:.4f}) to {ckpt_path}")

    final_train_acc = (train_acc_sum / train_acc_count) if train_acc_count > 0 else last_train_acc
    csv_rows.append({
        "framework": FRAMEWORK, "task": dataset_name,
        "norm": norm, "optimizer": opt_name, "loss": loss_name,
        "flags": flag_tag(flags),
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
    # ── projection-algorithm ablation flags (defaults = legacy behavior) ────
    parser.add_argument("--prenorm", choices=["none", "rms", "dyt"], default="none")
    parser.add_argument("--branch-mode", choices=["mean", "delta_sum"], default="mean")
    parser.add_argument("--softmax-mode", choices=["legacy", "rel_floor", "simplex_l2"], default="legacy")
    parser.add_argument("--qk-match-g", action="store_true",
                        help="Match g to omega in the QK^T pairwise projection")
    parser.add_argument("--rmsnorm-mode", choices=["legacy", "exact"], default="legacy")
    parser.add_argument("--residual-mode", choices=["full", "half"], default="full")
    parser.add_argument("--muon-acts", action="store_true",
                        help="Enable Muon orthogonalization on activation targets")
    parser.add_argument("--muon-acts-lr", type=float, default=1.0)
    parser.add_argument("--muon-acts-eps", type=float, default=1e-2)
    parser.add_argument("--muon-acts-mode",
                        choices=["fixed", "rel_row", "rel_frob", "raw_rel_row"],
                        default="fixed",
                        help="How to size the activation-target update once "
                             "direction is fixed (see core/ops.process_activation_target)")
    parser.add_argument("--muon-acts-max-dim", type=int, default=4096,
                        help="Skip rescale for activations with last_dim > this "
                             "(default 4096 excludes the flatten head; 0 disables)")
    parser.add_argument("--muon-acts-decay", choices=["none", "cosine", "linear"],
                        default="none",
                        help="Anneal muon_acts_lr to 0 over --muon-acts-decay-steps "
                             "(early-training bootstrap; constant lr compounds noise late)")
    parser.add_argument("--muon-acts-decay-steps", type=int, default=0,
                        help="Decay horizon in steps (0 = max_steps)")
    parser.add_argument("--momentum", type=float, default=0.95,
                        help="ProjectionMuonV2 momentum (default 0.95)")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="ProjectionMuonV2 decoupled weight decay (AdamW ref uses 0.01)")
    parser.add_argument("--ce-lambda", type=float, default=1.0,
                        help="CrossEntropyProjection proximal step size (config.cross_entropy_lambda)")
    parser.add_argument("--ce-num-steps", type=int, default=5,
                        help="CrossEntropyProjection inner iterations (config.cross_entropy_num_steps)")
    parser.add_argument("--save-best", action="store_true",
                        help="Save the best-val state dict to results/ckpt_<tag>_run<n>.pt "
                             "(for probe_head.py; not part of the flag tag)")
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"],
                        default="constant",
                        help="Optimizer lr schedule (cosine anneals to 0 over max_steps)")
    parser.add_argument("--proj-alpha", default=1.0,
                        type=lambda s: s if s == "auto" else float(s),
                        help="Weight-movement penalty of bilinear solves "
                             "(config.projection_alpha); 'auto' balances "
                             "qa/qb per layer")
    parser.add_argument("--proj-g", type=float, default=1.0,
                        help="Target-fidelity weight of bilinear solves (config.projection_g)")
    parser.add_argument("--head-pool", choices=["flatten", "avg", "max"], default="flatten")
    parser.add_argument("--td-lambda", type=float, default=0.0,
                        help="Direction-pipe trace strength (0 = off, exact legacy backward)")
    parser.add_argument("--td-mode", choices=["norm", "vector"], default="norm")
    parser.add_argument("--td-deflate", action="store_true",
                        help="Deflate the activation-parallel artifact from the trace residual")
    parser.add_argument("--td-eps-lin", type=float, default=1e-2,
                        help="Linear-transport amplitude cap (fraction of ||A||)")
    args = parser.parse_args()

    flags = {
        "prenorm": args.prenorm,
        "branch_mode": args.branch_mode,
        "softmax_mode": args.softmax_mode,
        "qk_match_g": args.qk_match_g,
        "rmsnorm_mode": args.rmsnorm_mode,
        "residual_mode": args.residual_mode,
        "muon_acts": args.muon_acts,
        "muon_acts_lr": args.muon_acts_lr,
        "muon_acts_eps": args.muon_acts_eps,
        "muon_acts_mode": args.muon_acts_mode,
        "muon_acts_max_dim": args.muon_acts_max_dim,
        "muon_acts_decay": args.muon_acts_decay,
        "muon_acts_decay_steps": args.muon_acts_decay_steps,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "ce_lambda": args.ce_lambda,
        "ce_num_steps": args.ce_num_steps,
        "lr_schedule": args.lr_schedule,
        "proj_alpha": args.proj_alpha,
        "proj_g": args.proj_g,
        "head_pool": args.head_pool,
        "td_lambda": args.td_lambda,
        "td_mode": args.td_mode,
        "td_deflate": args.td_deflate,
        "td_eps_lin": args.td_eps_lin,
    }

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
                        print(f"\n{'='*60}\n{FRAMEWORK} | {task_cfg['name']} | norm={norm} | opt={opt_name} | loss={loss_name} | flags={flag_tag(flags)}")
                        for run_number in range(1, cfg["num_runs"] + 1):
                            run(cfg, task_cfg, batch_size, run_number, device,
                                norm=norm, opt_name=opt_name, opt_kwargs=opt_kwargs, loss_name=loss_name,
                                flags=flags, save_best=args.save_best)
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
