##################################################
###   Benchmark — torch (PyTorch baseline)    ###
###   CNN architecture                        ###
##################################################
import sys, os, argparse
import gc
import yaml
import torch
import torch.nn.functional as F
import pandas as pd
import tqdm, time

from experiments.cnn_benchmarks.models import SimpleCNN_Torch
from experiments.shared.data import MNISTDataModule, InfiniteCifarDataModule

FRAMEWORK = "torch"

DATASETS = {"MNIST": MNISTDataModule, "CIFAR10": InfiniteCifarDataModule}

# ── Training ─────────────────────────────────────────────────────────────────

def run(cfg, task_cfg, batch_size, run_number, device,
        norm='l2', opt_name=None, opt_kwargs=None, loss_name='CrossEntropy', use_muon_activations=False):
    seed = cfg["random_seed"]
    run_seed = seed + run_number
    torch.manual_seed(run_seed)

    dataset_name = task_cfg.get("dataset", task_cfg["name"])
    dataset_cls = DATASETS[dataset_name]
    
    if dataset_name == "CIFAR10":
        ds = dataset_cls(batch_size=batch_size, data_dir="./dataset", seed=run_seed)
    else:
        ds = dataset_cls(batch_size=batch_size, seed=run_seed)
        
    train_iter = ds.train_iterator()
    val_loader  = ds.val_dataloader()
    test_loader = ds.test_dataloader()

    model = SimpleCNN_Torch(
        classes=task_cfg["classes"],
        in_channels=task_cfg.get("in_channels", 3)
    ).to(device)

    opt_name   = opt_name   or cfg["torch_optimizer"]
    opt_kwargs = opt_kwargs if opt_kwargs is not None else cfg.get("torch_optimizer_kwargs", {})
    optimizer  = getattr(torch.optim, opt_name)(model.parameters(), **opt_kwargs)

    # Build a tag that distinguishes this run in the CSV
    tag = f"{FRAMEWORK}_norm={norm}_opt={opt_name}_loss={loss_name}_muon={use_muon_activations}"

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
                if xv.shape[-1] in [1, 3]:  
                    xv = xv.permute(0, 3, 1, 2)
                yv = torch.tensor(yv, dtype=torch.long,  device=device)
                logits = model(xv)
                accs.append((logits.argmax(dim=-1) == yv).float().mean())
        model.train()
        return float(torch.stack(accs).mean())

    best_val_acc, best_state, best_step = 0.0, None, 0
    no_improve = 0
    t0 = time.time()

    step = 0
    with tqdm.tqdm(total=cfg["max_steps"], unit="step",
                   desc=f"{task_cfg['name']} | norm={norm} | {opt_name} | {loss_name}") as pbar:
        while step < cfg["max_steps"]:
            # ── Fetch a new batch ─────────────────────────────────────────────
            x_np, y_np = next(train_iter)
            x_batch = torch.tensor(x_np, dtype=torch.float32, device=device)
            if x_batch.shape[-1] in [1, 3]:
                x_batch = x_batch.permute(0, 3, 1, 2)
            y_batch = torch.tensor(y_np, dtype=torch.long, device=device)

            # ── Forward + Backward + Step ─────────────────────────────────────
            model.train()
            logits = model(x_batch)
            loss = F.cross_entropy(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ── Eval ──────────────────────────────────────────────────────────
            if step % cfg["eval_every"] == 0:
                val_acc = eval_acc(val_loader)
                elapsed = time.time() - t0
                csv_rows.append({
                    "framework": FRAMEWORK, "task": task_cfg["name"],
                    "norm": norm, "optimizer": opt_name, "loss": loss_name,
                    "use_muon_activations": use_muon_activations,
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
            
            if no_improve >= cfg["patience"]:
                break

    total_time = time.time() - t0
    if best_state:
        model.load_state_dict(best_state)

    test_acc = eval_acc(test_loader)
    print(f"Test Acc: {test_acc:.4f}  Time: {total_time:.1f}s")
    csv_rows.append({
        "framework": FRAMEWORK, "task": task_cfg["name"],
        "norm": norm, "optimizer": opt_name, "loss": loss_name,
        "use_muon_activations": use_muon_activations,
        "run": run_number, "step": step,
        "val_acc": test_acc, "wall_time_s": total_time,
    })

    # Write / append CSV
    df = pd.DataFrame(csv_rows)
    df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)
    print(f"Results appended to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Torch CNN benchmark")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"),
                        help="Path to YAML config file")
    parser.add_argument("--max-steps", type=int, default=None, help="Override cfg['max_steps']")
    parser.add_argument("--num-runs", type=int, default=None, help="Override cfg['num_runs']")
    args = parser.parse_args()

    cfg    = yaml.safe_load(open(args.config))
    if args.max_steps is not None: cfg["max_steps"] = args.max_steps
    if args.num_runs is not None: cfg["num_runs"] = args.num_runs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Standardize values for plotting comparison
    norm = "l2"
    loss_name = "CrossEntropy"
    muon_act = False

    # Optimizer sweep fallback
    optimizers = cfg.get("optimizers", [
        {"name": cfg.get("torch_optimizer", "Adam"),
         "kwargs": cfg.get("torch_optimizer_kwargs", {})}
    ])

    for batch_size in cfg["batch_sizes"]:
        for task_cfg in cfg["tasks"]:
            for opt_entry in optimizers:
                opt_name   = opt_entry["name"]
                opt_kwargs = opt_entry.get("kwargs", {})
                header = (f"{FRAMEWORK} | {task_cfg['name']} "
                          f"| norm={norm} | opt={opt_name} | loss={loss_name} | muon={muon_act} | bs={batch_size}")
                print(f"\n{'='*60}\n{header}")
                for run_number in range(1, cfg["num_runs"] + 1):
                    run(cfg, task_cfg, batch_size, run_number, device,
                        norm=norm, opt_name=opt_name, opt_kwargs=opt_kwargs, loss_name=loss_name, use_muon_activations=muon_act)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
