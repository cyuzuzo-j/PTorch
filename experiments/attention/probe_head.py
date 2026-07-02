##################################################
###   Linear-probe diagnostic for the ViT body ###
##################################################
"""Tests FINDINGS hypothesis #2 (the head solve is the remaining ceiling).

Loads a projection-trained ViT body from a bench checkpoint (produced with
`bench_ptorch_vit.py --save-best`), freezes it, extracts the exact features the
head consumes (flattened B×(T·D) for head_pool=flatten), and trains a plain
torch Linear readout with AdamW on them.

Decision rule:
  probe val ≈ AdamW reference (0.70) → body features are fine, the projection
      head solve is the bottleneck → invest in the head/CE workstream.
  probe val ≈ bench val (0.58)      → body features are the limit → invest in
      body-signal work.
  --random-body control should land ≈ 0.50 (the linear-probe-on-random-features
      reading of the original baseline plateau).

Note: in this codebase val_dataloader and test_dataloader are the same 10k
CIFAR test split, so probe val compares directly against bench val numbers.
"""
import argparse
import os
import time

import pandas as pd
import torch
import torch.nn as nn

from experiments.attention.bench_ptorch_vit import (
    ViT_PTorch, FLAG_DEFAULTS, flag_tag, apply_global_flags,
)
from experiments.shared.data import InfiniteCifarLoader, FiniteCifarLoader

# Architecture of the current winner arm; used only for --random-body.
RANDOM_BODY_FLAGS = {**FLAG_DEFAULTS, "prenorm": "dyt", "qk_match_g": True}
CIFAR_TASK_CFG = {"name": "CIFAR10", "patch_size": 4, "emb_dim": 128,
                  "depth": 4, "num_heads": 16, "mlp_ratio": 4.0}


def build_model(args, device):
    if args.random_body:
        flags, task_cfg, norm = dict(RANDOM_BODY_FLAGS), dict(CIFAR_TASK_CFG), "l2"
        tag = "random_body"
        torch.manual_seed(args.seed)
    else:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        flags = {**FLAG_DEFAULTS, **ckpt["flags"]}
        task_cfg, norm = ckpt["task_cfg"], ckpt["norm"]
        tag = flag_tag(flags)
        print(f"checkpoint: best_step={ckpt.get('best_step')} "
              f"best_val={ckpt.get('best_val_acc'):.4f}  flags={tag}")

    # Linear bakes alpha/g at construction time — apply before building.
    apply_global_flags(flags)
    model = ViT_PTorch(
        img_size=32, in_chans=3,
        patch_size=task_cfg.get("patch_size", 4),
        emb_dim=task_cfg.get("emb_dim", 128),
        depth=task_cfg.get("depth", 4),
        num_heads=task_cfg.get("num_heads", 16),
        mlp_ratio=task_cfg.get("mlp_ratio", 4.0),
        norm=norm,
        prenorm=flags["prenorm"],
        qk_match_g_to_omega=bool(flags["qk_match_g"]),
        residual_mode=flags["residual_mode"],
        head_pool=flags["head_pool"],
    ).to(device)
    if not args.random_body:
        model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tag


@torch.no_grad()
def extract_features(model, loader, device):
    feats, labels = [], []
    for x_np, y_np in loader:
        x = torch.tensor(x_np, dtype=torch.float32, device=device)
        if x.ndim == 4 and x.shape[-1] in [1, 3]:
            x = x.permute(0, 3, 1, 2)
        feats.append(model.forward_features(x).cpu())
        labels.append(torch.tensor(y_np, dtype=torch.long))
    return torch.cat(feats), torch.cat(labels)


def main():
    parser = argparse.ArgumentParser(description="Linear probe on a frozen projection-trained ViT body")
    parser.add_argument("--checkpoint", default=None,
                        help="ckpt_*.pt from bench_ptorch_vit.py --save-best")
    parser.add_argument("--random-body", action="store_true",
                        help="Control: probe an untrained body (winner architecture)")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--data-dir", default="./dataset")
    args = parser.parse_args()
    if (args.checkpoint is None) == (not args.random_body):
        parser.error("pass exactly one of --checkpoint or --random-body")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tag = build_model(args, device)

    # Un-augmented single pass over the 50k train split + the 10k test split
    # (the bench's "val" — see module docstring).
    train_loader = FiniteCifarLoader(
        InfiniteCifarLoader(args.data_dir, train=True, batch_size=args.batch_size), 50000)
    val_loader = FiniteCifarLoader(
        InfiniteCifarLoader(args.data_dir, train=False, batch_size=args.batch_size), 10000)

    t0 = time.time()
    train_x, train_y = extract_features(model, train_loader, device)
    val_x, val_y = extract_features(model, val_loader, device)
    print(f"features: train {tuple(train_x.shape)}  val {tuple(val_x.shape)}  "
          f"({time.time()-t0:.1f}s)")
    val_x, val_y = val_x.to(device), val_y.to(device)

    torch.manual_seed(args.seed)
    probe = nn.Linear(train_x.shape[1], 10).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def val_acc():
        probe.eval()
        accs = []
        for i in range(0, val_x.shape[0], 1024):
            logits = probe(val_x[i:i+1024])
            accs.append((logits.argmax(dim=-1) == val_y[i:i+1024]).float().mean())
        probe.train()
        return float(torch.stack(accs).mean())

    rows, best_val, best_step = [], 0.0, 0
    gen = torch.Generator().manual_seed(args.seed)
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, train_x.shape[0], (args.batch_size,), generator=gen)
        xb, yb = train_x[idx].to(device), train_y[idx].to(device)
        optimizer.zero_grad()
        logits = probe(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        if step % args.eval_every == 0 or step == args.steps:
            train_acc = float((logits.argmax(dim=-1) == yb).float().mean())
            va = val_acc()
            if va > best_val:
                best_val, best_step = va, step
            rows.append({"tag": tag, "step": step,
                         "train_acc": train_acc, "val_acc": va})
            print(f"step {step:5d}  train {train_acc:.4f}  val {va:.4f}  "
                  f"best {best_val:.4f}")

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"probe_head_{tag}.csv")
    pd.DataFrame(rows).to_csv(csv_path, mode="a",
                              header=not os.path.exists(csv_path), index=False)
    print(f"\nbest probe val_acc {best_val:.4f} @ step {best_step}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
