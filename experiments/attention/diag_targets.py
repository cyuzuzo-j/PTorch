##################################################
###   Diagnostics — projection target health   ###
###   for the ptorch ViT benchmark             ###
##################################################
"""Measures, over a short training run, where the projection learning signal
degrades in the ViT:

  * per-layer relative target delta ||t - a|| / ||a|| (vanishing targets show
    up as a geometric decay of this quantity from the head down to the patch
    embedding),
  * SoftmaxProjection clamp fraction and max logit displacement (numerical
    blowup from negative consensus targets),
  * matmul_proj Lagrange-multiplier |t| statistics and saturation (keyed by
    tensor shape: QK^T, attn@V, and 2-D Linear solves),
  * RMSNormProjection relative delta and sigma sign-flip fraction.

Accepts the same ablation flags as bench_ptorch_vit.py so each fix can be
re-measured. Results print as tables and are appended to
results/diag_targets_<flags>.csv.
"""
import argparse
import csv
import math
import os

import torch
import torch.nn.functional as F

from experiments.attention.bench_ptorch_vit import (
    ViT_PTorch, FLAG_DEFAULTS, flag_tag, apply_global_flags,
)
from experiments.shared.data import InfiniteCifarDataModule
from ptorch.nn.modules import CrossEntropy
from ptorch import config as ptorch_config
import ptorch.core.ops as ops
import ptorch.optim_static as ptorch_optim_static


class Recorder:
    """Running mean/max accumulator keyed by (record key, metric name)."""

    def __init__(self):
        self.stats = {}

    def record(self, key, **scalars):
        for metric, value in scalars.items():
            if not math.isfinite(value):
                value = float("nan")
            entry = self.stats.setdefault((key, metric), [0, 0.0, -float("inf")])
            entry[0] += 1
            entry[1] += value
            entry[2] = max(entry[2], value)

    def rows(self):
        for (key, metric), (count, total, peak) in sorted(self.stats.items()):
            yield key, metric, count, total / max(count, 1), peak

    def dump(self, path, step_bucket):
        new_file = not os.path.exists(path)
        with open(path, "a", newline="") as fh:
            writer = csv.writer(fh)
            if new_file:
                writer.writerow(["step_bucket", "key", "metric", "count", "mean", "max"])
            for key, metric, count, mean, peak in self.rows():
                writer.writerow([step_bucket, key, metric, count, f"{mean:.6g}", f"{peak:.6g}"])

    def reset(self):
        self.stats = {}


def attach_target_probes(model, recorder):
    """Hook named submodules so the backward target t for each output is
    compared against the forward activation a: rel_delta = ||t-a||/||a||."""
    probe_points = {"patch_embed": model.patch_embed, "head": model.head}
    for i, block in enumerate(model.blocks):
        probe_points[f"block{i}"] = block
        probe_points[f"block{i}.attn"] = block.attn
        probe_points[f"block{i}.mlp"] = block.mlp

    def make_forward_hook(name):
        def forward_hook(module, inputs, output):
            if not torch.is_tensor(output) or not output.requires_grad:
                return
            activation = output.detach()

            def target_hook(t):
                rel = (t - activation).norm() / (activation.norm() + 1e-12)
                recorder.record(f"layer/{name}", rel_delta=float(rel))

            output.register_hook(target_hook)
        return forward_hook

    handles = []
    for name, module in probe_points.items():
        handles.append(module.register_forward_hook(make_forward_hook(name)))
    return handles


def print_summary(recorder, step):
    print(f"\n──── diagnostics @ step {step} " + "─" * 40)
    layer_rows = [(k, m, c, mean, peak) for k, m, c, mean, peak in recorder.rows()
                  if k.startswith("layer/")]
    if layer_rows:
        print(f"{'layer':<22}{'mean rel_delta':>16}{'max rel_delta':>16}")
        for key, _metric, _count, mean, peak in layer_rows:
            print(f"{key[6:]:<22}{mean:>16.3e}{peak:>16.3e}")
    other_rows = [(k, m, c, mean, peak) for k, m, c, mean, peak in recorder.rows()
                  if not k.startswith("layer/")]
    if other_rows:
        print(f"\n{'key':<32}{'metric':<18}{'mean':>12}{'max':>12}")
        for key, metric, _count, mean, peak in other_rows:
            print(f"{key:<32}{metric:<18}{mean:>12.3e}{peak:>12.3e}")


def main():
    parser = argparse.ArgumentParser(description="PTorch ViT target diagnostics")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--prenorm", choices=["none", "rms", "dyt"], default="none")
    parser.add_argument("--branch-mode", choices=["mean", "delta_sum"], default="mean")
    parser.add_argument("--softmax-mode", choices=["legacy", "rel_floor", "simplex_l2"], default="legacy")
    parser.add_argument("--qk-match-g", action="store_true")
    parser.add_argument("--rmsnorm-mode", choices=["legacy", "exact"], default="legacy")
    parser.add_argument("--residual-mode", choices=["full", "half"], default="full")
    parser.add_argument("--muon-acts", action="store_true")
    parser.add_argument("--muon-acts-lr", type=float, default=1.0)
    parser.add_argument("--muon-acts-eps", type=float, default=1e-2)
    parser.add_argument("--muon-acts-mode",
                        choices=["fixed", "rel_row", "rel_frob", "raw_rel_row"],
                        default="fixed")
    parser.add_argument("--muon-acts-max-dim", type=int, default=4096)
    parser.add_argument("--muon-acts-decay", choices=["none", "cosine", "linear"],
                        default="none",
                        help="Mirrored from bench (no-op at diagnostic step counts)")
    parser.add_argument("--muon-acts-decay-steps", type=int, default=0)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--ce-lambda", type=float, default=1.0)
    parser.add_argument("--ce-num-steps", type=int, default=5)
    parser.add_argument("--proj-alpha", default=1.0,
                        type=lambda s: s if s == "auto" else float(s))
    parser.add_argument("--proj-g", type=float, default=1.0)
    parser.add_argument("--head-pool", choices=["flatten", "avg", "max"], default="flatten")
    args = parser.parse_args()

    flags = {**FLAG_DEFAULTS,
             "prenorm": args.prenorm, "branch_mode": args.branch_mode,
             "softmax_mode": args.softmax_mode, "qk_match_g": args.qk_match_g,
             "rmsnorm_mode": args.rmsnorm_mode, "residual_mode": args.residual_mode,
             "muon_acts": args.muon_acts, "muon_acts_lr": args.muon_acts_lr,
             "muon_acts_eps": args.muon_acts_eps,
             "muon_acts_mode": args.muon_acts_mode,
             "muon_acts_max_dim": args.muon_acts_max_dim,
             "muon_acts_decay": args.muon_acts_decay,
             "muon_acts_decay_steps": args.muon_acts_decay_steps,
             "momentum": args.momentum, "weight_decay": args.weight_decay,
             "ce_lambda": args.ce_lambda, "ce_num_steps": args.ce_num_steps,
             "proj_alpha": args.proj_alpha, "proj_g": args.proj_g,
             "head_pool": args.head_pool}
    apply_global_flags(flags)
    tag = flag_tag(flags)
    print(f"flags: {tag}")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = InfiniteCifarDataModule(batch_size=args.batch_size, seed=args.seed)
    train_iter = ds.train_iterator()

    model = ViT_PTorch(
        img_size=32, in_chans=3, patch_size=4, emb_dim=128, depth=4,
        num_heads=16, mlp_ratio=4.0, norm="l2",
        prenorm=flags["prenorm"],
        qk_match_g_to_omega=bool(flags["qk_match_g"]),
        residual_mode=flags["residual_mode"],
        head_pool=flags["head_pool"],
    ).to(device)
    optimizer = ptorch_optim_static.ProjectionMuonV2(
        model.parameters(), lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = CrossEntropy()

    recorder = Recorder()
    ops.DIAG = recorder
    attach_target_probes(model, recorder)

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"diag_targets_{tag}.csv")

    try:
        for step in range(1, args.steps + 1):
            x_np, y_np = next(train_iter)
            x = torch.tensor(x_np, dtype=torch.float32, device=device)
            if x.ndim == 4 and x.shape[-1] in [1, 3]:
                x = x.permute(0, 3, 1, 2)
            y = torch.tensor(y_np, dtype=torch.long, device=device)
            y_oh = F.one_hot(y, num_classes=10).float()

            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y_oh)
            loss.sum().backward()
            optimizer.step()

            if step % args.log_every == 0 or step == args.steps:
                train_acc = float((preds.argmax(dim=-1) == y).float().mean())
                print(f"\nstep {step}  train_acc {train_acc:.4f}  loss {float(loss.sum()):.4f}")
                print_summary(recorder, step)
                recorder.dump(csv_path, step)
                recorder.reset()
    finally:
        ops.DIAG = None

    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
