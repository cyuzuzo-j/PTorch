##################################################
###   Non-differentiable activations benchmark  ###
###   ReLU vs Step vs GappedStep vs QuantizedRelu#
###                                              #
###   Proves the ptorch projection backend can   #
###   train MLPs whose activations have no       #
###   subgradient (Step, GappedStep, QuantizedRelu#
###   are piecewise-constant) by routing through #
###   per-activation projections instead of      #
###   chain-rule autograd.                       #
##################################################
import sys, os, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))
import gc
import time
import yaml
import torch
import torch.nn as tnn
import torch.nn.functional as F
import pandas as pd
import tqdm

from ptorch.nn.modules import (
    Linear, ReLU, Step, GappedStep, QuantizedRelu,
    CrossEntropy,
)
import ptorch.nn.modules as ptorch_modules
from ptorch import config as ptorch_config
import ptorch.optim_static as ptorch_optim_static
from experiments.shared.data import MNISTDataModule, InfiniteCifarDataModule

ptorch_config.use_projections = True

FRAMEWORK = "ptorch"
OPTIM_MODULES = vars(ptorch_optim_static)
DATASETS = {"MNIST": MNISTDataModule, "CIFAR10": InfiniteCifarDataModule}

# Map activation name -> (factory(norm, **kwargs) -> module, is_differentiable)
ACTIVATIONS = {
    "ReLU":         (lambda norm, **kw: ReLU(norm=norm),     True),
    "Step":         (lambda norm, **kw: Step(),              False),
    "GappedStep":   (lambda norm, **kw: GappedStep(**kw),    False),
    "QuantizedRelu":(lambda norm, **kw: QuantizedRelu(**kw), False),
}


# ── Model ────────────────────────────────────────────────────────────────────

class MLP(tnn.Module):
    def __init__(self, hidden, in_features, classes, activation_name, norm='l2', activation_kwargs=None):
        super().__init__()
        activation_kwargs = activation_kwargs or {}
        act_factory, _ = ACTIVATIONS[activation_name]
        self.activation_name = activation_name
        self.layers = tnn.ModuleList()
        last = in_features
        for f in hidden:
            self.layers.append(Linear(last, f, norm=norm))
            self.layers.append(act_factory(norm, **activation_kwargs))
            last = f
        self.out = Linear(last, classes, norm=norm)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        for layer in self.layers:
            x = layer(x)
        return self.out(x)


# ── Training ─────────────────────────────────────────────────────────────────

def run(cfg, task_cfg, batch_size, run_number, device,
        activation_name, activation_kwargs=None, norm='l2', opt_name=None, opt_kwargs=None,
        loss_name='CrossEntropy'):
    seed = cfg["random_seed"]
    run_seed = seed + run_number
    torch.manual_seed(run_seed)

    dataset_name = task_cfg.get("dataset", task_cfg["name"])
    ds = DATASETS[dataset_name](batch_size=batch_size, seed=run_seed)
    train_iter = ds.train_iterator()
    val_loader = ds.val_dataloader()
    test_loader = ds.test_dataloader()

    model = MLP(task_cfg["hidden"], task_cfg["in_features"], task_cfg["classes"],
                activation_name=activation_name, norm=norm,
                activation_kwargs=activation_kwargs or {}).to(device)

    opt_name = opt_name or cfg["ptorch_optimizer"]
    opt_kwargs = opt_kwargs if opt_kwargs is not None else cfg.get("ptorch_optimizer_kwargs", {})
    optimizer = OPTIM_MODULES[opt_name](model.parameters(), **opt_kwargs)
    loss_fn = getattr(ptorch_modules, loss_name, CrossEntropy)()

    is_diff = ACTIVATIONS[activation_name][1]
    tag = f"{FRAMEWORK}_act={activation_name}_norm={norm}_opt={opt_name}_loss={loss_name}"

    results_dir = os.path.join(os.path.dirname(__file__), cfg.get("results_dir", "results"))
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"{tag}_{task_cfg['name']}.csv")
    csv_rows = []

    def eval_acc(loader):
        model.eval()
        accs = []
        with torch.no_grad():
            for xv, yv in loader:
                xv = torch.tensor(xv, dtype=torch.float32, device=device)
                yv = torch.tensor(yv, dtype=torch.long, device=device)
                logits = model.forward(xv)
                accs.append((logits.argmax(dim=-1) == yv).float().mean())
        model.train()
        return float(torch.stack(accs).mean())

    best_val_acc, best_state, best_step = 0.0, None, 0
    no_improve = 0
    t0 = time.time()

    step = 0
    pbar_desc = f"{task_cfg['name']} | act={activation_name} | diff={is_diff} | {opt_name}"
    with tqdm.tqdm(total=cfg["max_steps"], unit="step", desc=pbar_desc) as pbar:
        while step < cfg["max_steps"]:
            x_np, y_np = next(train_iter)
            x_batch = torch.tensor(x_np, dtype=torch.float32, device=device)
            y_batch = torch.tensor(y_np, dtype=torch.long, device=device)
            y_oh = F.one_hot(y_batch, num_classes=task_cfg["classes"]).float()

            output = model(x_batch)
            loss_fn(output, y_oh).sum().backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % cfg["eval_every"] == 0:
                val_acc = eval_acc(val_loader)
                elapsed = time.time() - t0
                csv_rows.append({
                    "framework": FRAMEWORK, "task": task_cfg["name"],
                    "activation": activation_name, "differentiable": is_diff,
                    "norm": norm, "optimizer": opt_name, "loss": loss_name,
                    "run": run_number, "step": step,
                    "val_acc": val_acc, "wall_time_s": elapsed,
                })
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

            pbar.update(1)
            step += 1

    total_time = time.time() - t0
    if best_state:
        model.load_state_dict(best_state)

    test_acc = eval_acc(test_loader)
    print(f"[{activation_name}] Test Acc: {test_acc:.4f}  Best Val: {best_val_acc:.4f}  Time: {total_time:.1f}s")
    csv_rows.append({
        "framework": FRAMEWORK, "task": task_cfg["name"],
        "activation": activation_name, "differentiable": is_diff,
        "norm": norm, "optimizer": opt_name, "loss": loss_name,
        "run": run_number, "step": step,
        "val_acc": test_acc, "wall_time_s": total_time,
        "split": "test",
    })

    df = pd.DataFrame(csv_rows)
    df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)
    print(f"Results appended to {csv_path}")
    return test_acc, best_val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Non-differentiable activations benchmark")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"),
                        help="Path to YAML config file")
    parser.add_argument("--activations", nargs="*", default=None,
                        help="Subset of activations to run (default: all four)")
    parser.add_argument("--max-steps", type=int, default=None, help="Override cfg['max_steps']")
    parser.add_argument("--num-runs", type=int, default=None, help="Override cfg['num_runs']")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.max_steps is not None: cfg["max_steps"] = args.max_steps
    if args.num_runs is not None: cfg["num_runs"] = args.num_runs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    norms = cfg.get("norms", ["l2"])
    losses = cfg.get("losses", [cfg.get("ptorch_loss", "CrossEntropy")])
    optimizers = cfg.get("optimizers", [])
    if not optimizers:
        optimizers = [{"name": cfg.get("ptorch_optimizer", "ProjectionAdam"),
                       "kwargs": cfg.get("ptorch_optimizer_kwargs", {})}]
    activations = args.activations or cfg.get("activations", list(ACTIVATIONS.keys()))
    activation_kwargs_map = cfg.get("activation_kwargs", {})  # name -> dict

    summary = []
    for batch_size in cfg["batch_sizes"]:
        for task_cfg in cfg["tasks"]:
            for activation_name in activations:
                for norm in norms:
                    for opt_entry in optimizers:
                        for loss_name in losses:
                            opt_name = opt_entry["name"]
                            opt_kwargs = opt_entry.get("kwargs", {})
                            header = (f"{FRAMEWORK} | {task_cfg['name']} | act={activation_name} "
                                      f"| norm={norm} | opt={opt_name} | loss={loss_name} | bs={batch_size}")
                            print(f"\n{'='*70}\n{header}")
                            act_kwargs = activation_kwargs_map.get(activation_name, {})
                            for run_number in range(1, cfg["num_runs"] + 1):
                                test_acc, best_val = run(
                                    cfg, task_cfg, batch_size, run_number, device,
                                    activation_name=activation_name,
                                    activation_kwargs=act_kwargs,
                                    norm=norm,
                                    opt_name=opt_name, opt_kwargs=opt_kwargs, loss_name=loss_name,
                                )
                                summary.append({
                                    "task": task_cfg["name"],
                                    "activation": activation_name,
                                    "differentiable": ACTIVATIONS[activation_name][1],
                                    "run": run_number,
                                    "test_acc": test_acc,
                                    "best_val_acc": best_val,
                                })
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("Summary (test accuracy by activation):")
    print("=" * 70)
    summary_df = pd.DataFrame(summary)
    if not summary_df.empty:
        agg = summary_df.groupby(["task", "activation", "differentiable"])["test_acc"].agg(["mean", "std", "count"])
        print(agg.to_string())
