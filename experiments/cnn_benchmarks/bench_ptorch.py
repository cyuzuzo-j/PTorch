##################################################
###   Benchmark — ptorch (cyclic projections)  ###
###   CNN architecture                         ###
##################################################
import sys, os, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))

import gc
import yaml
import torch
import torch.nn as tnn
import torch.nn.functional as F
import pandas as pd
import ptorch.nn.modules as pnn
from ptorch.nn.modules import (
    ProjectionModule, CrossEntropy, HardMarginLoss, ProximalHingeMarginLoss
)
from ptorch import config as ptorch_config
import ptorch.optim_static as ptorch_optim_static
from experiments.shared.data import MNISTDataModule, InfiniteCifarDataModule
import tqdm, time

ptorch_config.config.use_projections = True

FRAMEWORK = "ptorch_cnn"
OPTIM_MODULES = vars(ptorch_optim_static)

DATASETS = {"MNIST": MNISTDataModule, "CIFAR10": InfiniteCifarDataModule}

# ── Model ────────────────────────────────────────────────────────────────────

class SimpleCNN_PTorch(tnn.Module):
    def __init__(self, classes=10, in_channels=3, conv_features=32, alpha=1.0, g=1.0, loss_name='CrossEntropy'):
        super().__init__()
        
        self.conv1 = pnn.Conv2D(in_channels, conv_features, kernel_size=3, padding=1, alpha=alpha, g=g)
        self.pool1 = pnn.MaxPool2d(2)
        self.relu1 = pnn.LeakyReLU(0.1)

        self.conv2 = pnn.Conv2D(conv_features, conv_features * 2, kernel_size=3, padding=1, alpha=alpha, g=g)
        self.pool2 = pnn.MaxPool2d(2)
        self.relu2 = pnn.LeakyReLU(0.1)

        self.conv3 = pnn.Conv2D(conv_features * 2, conv_features * 4, kernel_size=3, padding=1, alpha=alpha, g=g)
        self.pool3 = pnn.MaxPool2d(2)
        self.relu3 = pnn.LeakyReLU(0.1)

        # Predict flattening size
        with torch.no_grad():
            dummy_res = 32 if in_channels == 3 else 28
            dummy_in = torch.zeros(1, in_channels, dummy_res, dummy_res)
            dummy_out = self.relu3(self.pool3(self.conv3(self.relu2(self.pool2(self.conv2(self.relu1(self.pool1(self.conv1(dummy_in)))))))))
            flatten_size = dummy_out.reshape(1, -1).size(1)
        self.head = pnn.Linear(flatten_size, classes, norm="inf")

    def forward(self, x):
        x = self.relu1(self.pool1(self.conv1(x)))
        x = self.relu2(self.pool2(self.conv2(x)))
        x = self.relu3(self.pool3(self.conv3(x)))
        x = x.flatten(1)
        return self.head(x)


# ── Training ─────────────────────────────────────────────────────────────────

def run(cfg, task_cfg, batch_size, run_number, device,
        norm='l2', opt_name=None, opt_kwargs=None, loss_name='CrossEntropy', use_muon_activations=False):
    """Single training run.

    Parameters
    ----------
    norm : str
        Projection norm passed to every Linear layer ('l2' or 'linf').
    opt_name : str | None
        Override optimizer class name (falls back to cfg['ptorch_optimizer']).
    opt_kwargs : dict | None
        Override optimizer kwargs (falls back to cfg['ptorch_optimizer_kwargs']).
    loss_name : str
        Name of the loss module to use.
    """
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

    model = SimpleCNN_PTorch(
        classes=task_cfg["classes"],
        in_channels=task_cfg.get("in_channels", 3),
    ).to(device)

    loss_cls = getattr(pnn, loss_name, CrossEntropy)
    criterion = loss_cls()

    ptorch_config.config.use_muon_activations = use_muon_activations

    opt_name   = opt_name   or cfg["ptorch_optimizer"]
    opt_kwargs = opt_kwargs if opt_kwargs is not None else cfg.get("ptorch_optimizer_kwargs", {})
    optimizer  = OPTIM_MODULES[opt_name](model.parameters(), **opt_kwargs)

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
            y_oh    = F.one_hot(y_batch, num_classes=task_cfg["classes"]).float()

            # ── Forward + Backward + Step ─────────────────────────────────────
            model.train()
            logits = model(x_batch)
            loss = criterion(logits, y_oh)
            optimizer.zero_grad()
            loss.sum().backward()
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
    parser = argparse.ArgumentParser(description="PTorch CNN benchmark")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"),
                        help="Path to YAML config file")
    args = parser.parse_args()

    cfg    = yaml.safe_load(open(args.config))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Sweep axes from config (default to single values when absent)
    norms = cfg.get("norms", ["l2"])
    losses = cfg.get("losses", [cfg.get("ptorch_loss", "CrossEntropy")])

    # Optimizer sweep: list of {name, kwargs} dicts, or fall back to the
    # single ptorch_optimizer / ptorch_optimizer_kwargs pair.
    optimizers = cfg.get("optimizers", [
        {"name": cfg["ptorch_optimizer"],
         "kwargs": cfg.get("ptorch_optimizer_kwargs", {})}
    ])

    muon_sweeps = cfg.get("use_muon_activations", [False])
    if not isinstance(muon_sweeps, list):
        muon_sweeps = [muon_sweeps]

    for batch_size in cfg["batch_sizes"]:
        for task_cfg in cfg["tasks"]:
            for norm in norms:
                for opt_entry in optimizers:
                    for loss_name in losses:
                        for muon_act in muon_sweeps:
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