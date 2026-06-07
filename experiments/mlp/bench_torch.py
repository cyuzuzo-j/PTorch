##################################################
###   Benchmark — torch (PyTorch baseline)    ###
###   Standard Adam + cross-entropy           ###
##################################################
import sys, os
import gc
import yaml
import torch
import torch.nn as tnn
import torch.nn.functional as F
import pandas as pd
from experiments.shared.data import MNISTDataModule, InfiniteCifarDataModule
import tqdm, time

FRAMEWORK = "torch"
CFG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

DATASETS = {"MNIST": MNISTDataModule, "CIFAR10": InfiniteCifarDataModule}


# ── Model ────────────────────────────────────────
class MLP(tnn.Module):
    def __init__(self, hidden, in_features, classes):
        super().__init__()
        last = in_features
        layers = []
        for f in hidden:
            layers += [tnn.Linear(last, f), tnn.ReLU()]
            last = f
        self.body = tnn.Sequential(*layers)
        self.out  = tnn.Linear(last, classes)

    def forward(self, x):
        return self.out(self.body(x.reshape(x.shape[0], -1)))


# ── Training ─────────────────────────────────────
def run(cfg, task_cfg, batch_size, run_number, device, opt_name=None, opt_kwargs=None):
    seed = cfg["random_seed"]
    run_seed = seed + run_number
    torch.manual_seed(run_seed)

    dataset_name = task_cfg.get("dataset", task_cfg["name"])
    dataset_cls = DATASETS[dataset_name]
    ds = dataset_cls(batch_size=batch_size, seed=run_seed)
    train_iter = ds.train_iterator()
    val_loader  = ds.val_dataloader()
    test_loader = ds.test_dataloader()

    model      = MLP(task_cfg["hidden"], task_cfg["in_features"], task_cfg["classes"]).to(device)
    opt_name   = opt_name or cfg["torch_optimizer"]
    opt_kwargs = opt_kwargs if opt_kwargs is not None else cfg.get("torch_optimizer_kwargs", {})
    optimizer  = getattr(torch.optim, opt_name)(model.parameters(), **opt_kwargs)

    run_name = (
        f"{cfg.get('experiment_name', 'run')}_{FRAMEWORK}"
        f"_{task_cfg['name']}_bs{batch_size}_run{run_number}_{opt_name}"
    )

    results_dir = os.path.join(os.path.dirname(__file__), cfg.get("results_dir", "results"))
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"{FRAMEWORK}_{task_cfg['name']}.csv")
    csv_rows = []

    def step_fn(x, y):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y.long())
        loss.backward()
        optimizer.step()
        return loss

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
                    torch.tensor(x, device=device, dtype=torch.float),
                    torch.tensor(y, device=device, dtype=torch.long))
                    for x, y in val_loader]
                val_acc = float(torch.stack(accs).mean())
                model.train()
                elapsed = time.time() - t0
                csv_rows.append({
                    "framework": FRAMEWORK, "task": task_cfg["name"],
                    "optimizer": opt_name, "run": run_number, "step": step,
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

            x, y = next(train_iter)
            step_fn(
                torch.tensor(x, device=device, dtype=torch.float),
                torch.tensor(y, device=device, dtype=torch.long)
            )
            step += 1
            pbar.update(1)
            if cfg["max_steps"] and step >= cfg["max_steps"]:
                break

    total_time = time.time() - t0
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    test_accs = [eval_fn(
        torch.tensor(x, device=device, dtype=torch.float),
        torch.tensor(y, device=device, dtype=torch.long))
        for x, y in test_loader]
    final_acc = float(torch.stack(test_accs).mean())
    print(f"Test Acc: {final_acc:.4f}  Time: {total_time:.1f}s")
    csv_rows.append({
        "framework": FRAMEWORK, "task": task_cfg["name"],
        "optimizer": opt_name, "run": run_number, "step": step,
        "val_acc": final_acc, "wall_time_s": total_time,
    })

    # Write / append CSV
    df = pd.DataFrame(csv_rows)
    df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)
    print(f"Results appended to {csv_path}")
    return final_acc, best_val_acc, best_step, total_time


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Torch MLP baseline")
    parser.add_argument("--config", default=CFG_PATH, help="Path to YAML config file")
    parser.add_argument("--max-steps", type=int, default=None, help="Override cfg['max_steps']")
    parser.add_argument("--num-runs", type=int, default=None, help="Override cfg['num_runs']")
    args = parser.parse_args()

    cfg    = yaml.safe_load(open(args.config))
    if args.max_steps is not None: cfg["max_steps"] = args.max_steps
    if args.num_runs is not None: cfg["num_runs"] = args.num_runs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    optimizers = cfg.get("optimizers", [
        {"name": cfg.get("torch_optimizer", "Adam"),
         "kwargs": cfg.get("torch_optimizer_kwargs", {})}
    ])

    for batch_size in cfg["batch_sizes"]:
        for task_cfg in cfg["tasks"]:
            for opt_entry in optimizers:
                opt_name = opt_entry["name"]
                opt_kwargs = opt_entry.get("kwargs", {})
                print(f"\n{'='*50}\n{FRAMEWORK} | {task_cfg['name']} | bs={batch_size} | opt={opt_name}")
                for run_number in range(1, cfg["num_runs"] + 1):
                    run(cfg, task_cfg, batch_size, run_number, device, opt_name=opt_name, opt_kwargs=opt_kwargs)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
