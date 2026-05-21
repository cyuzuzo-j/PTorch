##################################################
###   Hybrid training — CNN                   ###
###   K gradient steps ↔ K projection steps   ###
###   Optimizer state is reset on each switch  ###
##################################################
import sys, os, argparse, gc, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))

import yaml
import torch
import torch.nn.functional as F
import pandas as pd
import tqdm

import ptorch.nn.modules as pnn
from ptorch.nn.modules import CrossEntropy, HardMarginLoss
from ptorch import config as ptorch_config
import ptorch.optim_static as ptorch_optim_static
from experiments.cnn_benchmarks.models import SimpleCNN_PTorch
from experiments.shared.data import MNISTDataModule, InfiniteCifarDataModule

DATASETS    = {"MNIST": MNISTDataModule, "CIFAR10": InfiniteCifarDataModule}
OPTIM_MODULES = vars(ptorch_optim_static)
FRAMEWORK   = "hybrid_cnn"


# ── Optimizer factories ───────────────────────────────────────────────────────

def _make_grad_optimizer(model, name, kwargs):
    """Standard torch optimizer (no projection pseudo-gradients)."""
    cls = getattr(torch.optim, name)
    return cls(model.parameters(), **kwargs)


def _make_proj_optimizer(model, name, kwargs):
    """Projection optimizer that interprets p.grad as the projection target."""
    cls = OPTIM_MODULES[name]
    return cls(model.parameters(), **kwargs)


def _reset_proj_caches(model):
    """Drop all per-layer projection caches so stale targets don't bleed over."""
    for m in model.modules():
        if hasattr(m, "proj_cache"):
            m.proj_cache.clear()
        if hasattr(m, "projection_forward_cache"):
            m.projection_forward_cache[:] = [None] * len(m.projection_forward_cache)


# ── Single run ────────────────────────────────────────────────────────────────

def run(cfg, task_cfg, batch_size, run_number, device,
        K=50,
        grad_opt_name="Adam",    grad_opt_kwargs=None,
        proj_opt_name="ProjectionAdam", proj_opt_kwargs=None,
        loss_name="CrossEntropy",
        grad_clip=1.0):
    """
    Trains SimpleCNN_PTorch by alternating between:
      - K steps of standard gradient descent  (ptorch_config.config.use_projections = False)
      - K steps of cyclic projections          (ptorch_config.config.use_projections = True)

    The optimizer is fully discarded and re-created on every phase transition so
    momentum buffers, running statistics, etc., do not carry across modes.
    """
    grad_opt_kwargs = grad_opt_kwargs or {"lr": 1e-3}
    proj_opt_kwargs = proj_opt_kwargs or {"lr": 1e-3}

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
           f"_grad={grad_opt_name}_proj={proj_opt_name}_loss={loss_name}"
           f"_clip={grad_clip}")

    results_dir = os.path.join(os.path.dirname(__file__),
                               cfg.get("results_dir", "results"))
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"{tag}_{task_cfg['name']}.csv")

    # ── helpers ──────────────────────────────────────────────────────────────

    def to_nchw(x):
        """Move channel dim to front if batch arrived as NHWC."""
        if x.ndim == 4 and x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2)
        return x

    def eval_acc(loader):
        model.eval()
        # always eval with plain forward pass (no projection overhead)
        saved = ptorch_config.config.use_projections
        ptorch_config.config.use_projections = False
        accs = []
        with torch.no_grad():
            for xv, yv in loader:
                xv = to_nchw(torch.tensor(xv, dtype=torch.float32, device=device))
                yv = torch.tensor(yv, dtype=torch.long, device=device)
                accs.append((model(xv).argmax(dim=-1) == yv).float().mean())
        ptorch_config.config.use_projections = saved
        model.train()
        return float(torch.stack(accs).mean())

    # ── phase bootstrap ───────────────────────────────────────────────────────
    # Start in gradient phase
    phase      = "grad"
    phase_step = 0
    ptorch_config.config.use_projections = False
    optimizer  = _make_grad_optimizer(model, grad_opt_name, grad_opt_kwargs)

    csv_rows = []
    best_val_acc, best_state, best_step = 0.0, None, 0
    no_improve = 0
    t0 = time.time()

    with tqdm.tqdm(total=cfg["max_steps"], unit="step", desc=tag) as pbar:
        for step in range(cfg["max_steps"]):

            # ── Phase switch ──────────────────────────────────────────────
            if phase_step >= K:
                del optimizer
                _reset_proj_caches(model)

                if phase == "grad":
                    phase = "proj"
                    ptorch_config.config.use_projections = True
                    optimizer = _make_proj_optimizer(model, proj_opt_name, proj_opt_kwargs)
                else:
                    phase = "grad"
                    ptorch_config.config.use_projections = False
                    optimizer = _make_grad_optimizer(model, grad_opt_name, grad_opt_kwargs)

                phase_step = 0

            # ── Fetch batch ───────────────────────────────────────────────
            x_np, y_np = next(train_iter)
            x = to_nchw(torch.tensor(x_np, dtype=torch.float32, device=device))
            y = torch.tensor(y_np, dtype=torch.long, device=device)
            y_oh = F.one_hot(y, num_classes=task_cfg["classes"]).float()

            # ── Forward / backward / step ─────────────────────────────────
            model.train()
            logits = model(x)
            loss_val = criterion(logits, y_oh).sum()
            optimizer.zero_grad()
            loss_val.backward()
            if phase == "grad" and grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            phase_step += 1

            # ── Eval ──────────────────────────────────────────────────────
            if step % cfg["eval_every"] == 0:
                val_acc = eval_acc(val_loader)
                elapsed = time.time() - t0
                csv_rows.append({
                    "framework": FRAMEWORK, "task": task_cfg["name"],
                    "K": K, "grad_opt": grad_opt_name, "proj_opt": proj_opt_name,
                    "loss": loss_name, "run": run_number,
                    "step": step, "phase": phase,
                    "val_acc": val_acc, "wall_time_s": elapsed,
                })
                pbar.set_postfix(val_acc=f"{val_acc:.4f}", phase=phase,
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
        "K": K, "grad_opt": grad_opt_name, "proj_opt": proj_opt_name,
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
    parser.add_argument("--K",        type=int,   default=50,
                        help="Steps per phase before switching")
    parser.add_argument("--grad_opt", default="Adam",
                        help="torch.optim class for gradient phase")
    parser.add_argument("--grad_lr",  type=float, default=1e-3)
    parser.add_argument("--proj_opt", default="ProjectionAdam",
                        help="ptorch.optim_static class for projection phase")
    parser.add_argument("--proj_lr",  type=float, default=1e-3)
    parser.add_argument("--loss",     default="CrossEntropy")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Max gradient norm during gradient phase (0 = disabled)")
    args = parser.parse_args()

    cfg    = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"K={args.K}  grad={args.grad_opt}(lr={args.grad_lr})"
          f"  proj={args.proj_opt}(lr={args.proj_lr})")

    for batch_size in cfg["batch_sizes"]:
        for task_cfg in cfg["tasks"]:
            header = (f"hybrid_cnn | {task_cfg['name']}"
                      f" | K={args.K} | bs={batch_size}")
            print(f"\n{'='*60}\n{header}")
            for run_number in range(1, cfg["num_runs"] + 1):
                run(cfg, task_cfg, batch_size, run_number, device,
                    K=args.K,
                    grad_opt_name=args.grad_opt,
                    grad_opt_kwargs={"lr": args.grad_lr},
                    proj_opt_name=args.proj_opt,
                    proj_opt_kwargs={"lr": args.proj_lr},
                    loss_name=args.loss,
                    grad_clip=args.grad_clip if args.grad_clip > 0 else None)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
