##################################################
###   Hybrid training — CNN                   ###
###   K gradient steps ↔ K projection steps   ###
###   Single optimizer carries state across   ###
##################################################
import sys, os, argparse, copy, gc, time
from collections import defaultdict
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))

import yaml
import torch
import torch.nn.functional as F
import pandas as pd
import tqdm

import ptorch.nn.modules as pnn
from ptorch.nn.modules import CrossEntropy
from ptorch import config as ptorch_config
import ptorch.optim_static as ptorch_optim_static
from experiments.cnn_benchmarks.models import SimpleCNN_PTorch
from experiments.shared.data import MNISTDataModule, InfiniteCifarDataModule

DATASETS  = {"MNIST": MNISTDataModule, "CIFAR10": InfiniteCifarDataModule}
FRAMEWORK = "hybrid_cnn"


# ── Single run ────────────────────────────────────────────────────────────────

def run(cfg, task_cfg, batch_size, run_number, device,
        K=50,
        opt_name="ProjectionAdam", opt_kwargs=None,
        loss_name="CrossEntropy",
        grad_clip=None,
        grad_lr=1e-3, proj_lr=1e-3):
    """
    Trains SimpleCNN_PTorch by alternating between:
      - K steps of standard gradient descent  (ptorch_config.config.use_projections = False)
      - K steps of cyclic projections          (ptorch_config.config.use_projections = True)

    Adam state is partitioned per phase: the (m, v, step) buffers from the last
    grad block are restored when re-entering grad, and likewise for proj. This
    prevents grad-phase running statistics from contaminating the first proj
    steps (and vice-versa), since the two signals — true gradients ∂L/∂p and
    proximal-CE pseudo-gradients (p − p_proj) — live on different scales.
    """
    # Phase-dependent learning rates (initial phase is grad).
    opt_kwargs = dict(opt_kwargs or {})
    opt_kwargs["lr"] = grad_lr
    phase_lr = {"grad": grad_lr, "proj": proj_lr}

    seed = cfg["random_seed"] + run_number
    torch.manual_seed(seed)

    dataset_name = task_cfg.get("dataset", task_cfg["name"])
    ds_kwargs = {"batch_size": batch_size, "seed": seed}
    if dataset_name == "CIFAR10":
        ds_kwargs["data_dir"] = os.path.join(os.path.dirname(__file__), "dataset")
    ds = DATASETS[dataset_name](**ds_kwargs)

    train_iter  = ds.train_iterator()
    val_loader  = ds.val_dataloader()
    test_loader = ds.test_dataloader()

    model = SimpleCNN_PTorch(
        classes=task_cfg["classes"],
        in_channels=task_cfg.get("in_channels", 3),
    ).to(device)

    criterion = getattr(pnn, loss_name, CrossEntropy)()

    tag = (f"{FRAMEWORK}_K={K}"
           f"_opt={opt_name}_loss={loss_name}")

    results_dir = os.path.join(os.path.dirname(__file__),
                               cfg.get("results_dir", "results"))
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"{tag}_{task_cfg['name']}.csv")

    def to_nchw(x):
        if x.ndim == 4 and x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2)
        return x

    def eval_acc(loader):
        model.eval()
        saved = ptorch_config.config.use_projections
        ptorch_config.config.use_projections = False
        try:
            accs = []
            with torch.no_grad():
                for xv, yv in loader:
                    xv = to_nchw(torch.tensor(xv, dtype=torch.float32, device=device))
                    yv = torch.tensor(yv, dtype=torch.long, device=device)
                    accs.append((model(xv).argmax(dim=-1) == yv).float().mean())
            return float(torch.stack(accs).mean())
        finally:
            ptorch_config.config.use_projections = saved
            model.train()

    # ── Bootstrap ────────────────────────────────────────────────────────────
    phase, phase_step = "grad", 0
    ptorch_config.config.use_projections = False
    optimizer = getattr(ptorch_optim_static, opt_name)(model.parameters(), **opt_kwargs)
    # Per-phase Adam state snapshots — grad and proj keep independent (m, v, step)
    # buffers so neither phase's running statistics leak into the other.
    phase_opt_state = {"grad": None, "proj": None}

    csv_rows = []
    best_val_acc, best_state, best_step = 0.0, None, 0
    no_improve = 0
    t0 = time.time()

    with tqdm.tqdm(total=cfg["max_steps"], unit="step", desc=tag) as pbar:
        for step in range(cfg["max_steps"]):

            # ── Phase switch ──────────────────────────────────────────────
            if phase_step >= K:
                # Stash the current phase's Adam state, swap, and restore the
                # other phase's state (fresh zeros on its first entry).
                phase_opt_state[phase] = copy.deepcopy(optimizer.state_dict())
                phase = "proj" if phase == "grad" else "grad"
                ptorch_config.config.use_projections = (phase == "proj")
                if phase_opt_state[phase] is not None:
                    optimizer.load_state_dict(phase_opt_state[phase])
                else:
                    optimizer.state = defaultdict(dict)
                # Apply this phase's learning rate (lr=10 in proj, lr=0.01 in grad).
                for pg in optimizer.param_groups:
                    pg["lr"] = phase_lr[phase]
                phase_step = 0

            # ── Fetch batch ───────────────────────────────────────────────
            x_np, y_np = next(train_iter)
            x = to_nchw(torch.tensor(x_np, dtype=torch.float32, device=device))
            y = torch.tensor(y_np, dtype=torch.long, device=device)
            y_oh = F.one_hot(y, num_classes=task_cfg["classes"]).float()

            # ── Forward / backward / step ─────────────────────────────────
            model.train()
            logits = model(x)
            optimizer.zero_grad()
            criterion(logits, y_oh).sum().backward()

            if phase == "grad" and grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # ── Diagnostics: norm of what the optimizer actually consumes ──
            # In grad phase that's p.grad; in proj phase ProjectionAdam will
            # turn p.grad into (p - p_proj), so we report that here.
            with torch.no_grad():
                sq = 0.0
                for p in model.parameters():
                    if p.grad is None:
                        continue
                    d = (p.data - p.grad) if phase == "proj" else p.grad
                    sq += float(d.pow(2).sum())
                grad_norm = sq ** 0.5

            optimizer.step()
            phase_step += 1

            # ── Eval ──────────────────────────────────────────────────────
            if step % cfg["eval_every"] == 0:
                val_acc = eval_acc(val_loader)
                elapsed = time.time() - t0
                csv_rows.append({
                    "framework": FRAMEWORK, "task": task_cfg["name"],
                    "K": K, "opt": opt_name,
                    "loss": loss_name, "run": run_number,
                    "step": step, "phase": phase,
                    "val_acc": val_acc, "wall_time_s": elapsed,
                    "grad_norm": grad_norm,
                })
                pbar.set_postfix(val_acc=f"{val_acc:.4f}", phase=phase,
                                 gnorm=f"{grad_norm:.2e}",
                                 best=f"{best_val_acc:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc, best_step = val_acc, step
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve >= cfg["patience"]:
                    print(f"Early stop at step {step}")
                    break

            pbar.update(1)

    total_time = time.time() - t0
    if best_state:
        model.load_state_dict(best_state)

    test_acc = eval_acc(test_loader)
    print(f"[{tag}] Test={test_acc:.4f}  BestVal={best_val_acc:.4f} @step {best_step}"
          f"  Time={total_time:.1f}s")

    csv_rows.append({
        "framework": FRAMEWORK, "task": task_cfg["name"],
        "K": K, "opt": opt_name,
        "loss": loss_name, "run": run_number,
        "step": step, "phase": "test",
        "val_acc": test_acc, "wall_time_s": total_time,
    })

    df = pd.DataFrame(csv_rows)
    df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)
    print(f"Results appended to {csv_path}")
    return csv_rows


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid gradient/projection CNN training")
    parser.add_argument("--config",
                        default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    parser.add_argument("--K",        type=int,   default=200,
                        help="Steps per phase before switching")
    parser.add_argument("--opt",      default="ProjectionAdam",
                        help="ptorch.optim_static class used in both phases")
    parser.add_argument("--grad_lr",  type=float, default=1e-3,
                        help="Learning rate during gradient phase")
    parser.add_argument("--proj_lr",  type=float, default=1e-3,
                        help="Learning rate during projection phase")
    parser.add_argument("--loss",     default="CrossEntropy")
    parser.add_argument("--grad_clip", type=float, default=0.0,

                        help="Max gradient norm during grad phase (0 = disabled)")
    parser.add_argument("--max-steps", type=int, default=None, help="Override cfg['max_steps']")
    parser.add_argument("--num-runs", type=int, default=None, help="Override cfg['num_runs']")
    args = parser.parse_args()

    cfg    = yaml.safe_load(open(args.config))
    if args.max_steps is not None: cfg["max_steps"] = args.max_steps
    if args.num_runs is not None: cfg["num_runs"] = args.num_runs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"K={args.K}  opt={args.opt}(grad_lr={args.grad_lr}, proj_lr={args.proj_lr})")

    for batch_size in cfg["batch_sizes"]:
        for task_cfg in cfg["tasks"]:
            header = (f"hybrid_cnn | {task_cfg['name']}"
                      f" | K={args.K} | bs={batch_size}")
            print(f"\n{'='*60}\n{header}")
            for run_number in range(1, cfg["num_runs"] + 1):
                run(cfg, task_cfg, batch_size, run_number, device,
                    K=args.K,
                    opt_name=args.opt,
                    loss_name=args.loss,
                    grad_clip=args.grad_clip if args.grad_clip > 0 else None,
                    grad_lr=args.grad_lr, proj_lr=args.proj_lr)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
