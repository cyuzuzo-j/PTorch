##################################################
###   Benchmark — ptorch (cyclic projections)  ###
###   Supports sweeps over norms & optimizers  ###
##################################################
import sys, os, argparse
import gc
import yaml
import torch
torch.set_float32_matmul_precision('high')
import torch.nn as tnn
import torch.nn.functional as F
import pandas as pd
from ptorch.nn.modules import Linear, ReLU, HardMarginLoss
import ptorch.nn.modules as ptorch_modules
from ptorch import config as ptorch_config
import ptorch.optim_static as ptorch_optim_static
from data import MNISTDataModule, InfiniteCifarDataModule
import tqdm, time

ptorch_config.use_projections = True

FRAMEWORK = "ptorch"
OPTIM_MODULES = vars(ptorch_optim_static)

DATASETS = {"MNIST": MNISTDataModule, "CIFAR10": InfiniteCifarDataModule}


# ── Model ────────────────────────────────────────────────────────────────────

class MLP(tnn.Module):
    def __init__(self, hidden, in_features, classes, norm='l2', num_iters=1):
        super().__init__()
        last = in_features
        self.hidden_layers = tnn.ModuleList()

        for i, f in enumerate(hidden):
            self.hidden_layers.append(Linear(last, f, norm=norm, num_iters=num_iters))
            self.hidden_layers.append(ReLU( norm=norm))
            last = f
        self.out = Linear(last, classes, norm=norm, num_iters=num_iters)
        
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        for i in range(0, len(self.hidden_layers), 2):
            x = self.hidden_layers[i](x)
            x = self.hidden_layers[i + 1](x)
        return self.out(x)

# ── Training ─────────────────────────────────────────────────────────────────

def run(cfg, task_cfg, batch_size, run_number, device,
        norm='l2', opt_name=None, opt_kwargs=None, loss_name='HardMarginLoss', use_muon_activations=False,
        td_lambda=0.0, td_mode="norm", td_eps_lin=None, td_anneal=False,
        td_deflate=False, num_iters=1):
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
    ds = dataset_cls(batch_size=batch_size, seed=run_seed)
    train_iter = ds.train_iterator()
    val_loader  = ds.val_dataloader()
    test_loader = ds.test_dataloader()

    model = MLP(task_cfg["hidden"], task_cfg["in_features"],
                task_cfg["classes"], norm=norm, num_iters=num_iters).to(device)

    ptorch_config.config.use_muon_activations = use_muon_activations
    ptorch_config.config.update("td_lambda", float(td_lambda))
    ptorch_config.config.update("td_mode", td_mode)
    if td_eps_lin is not None:
        ptorch_config.config.update("td_eps_lin", float(td_eps_lin))
    ptorch_config.config.update("td_deflate", bool(td_deflate))

    opt_name   = opt_name   or cfg["ptorch_optimizer"]
    opt_kwargs = opt_kwargs if opt_kwargs is not None else cfg.get("ptorch_optimizer_kwargs", {})
    optimizer  = OPTIM_MODULES[opt_name](model.parameters(), **opt_kwargs)
    loss = getattr(ptorch_modules, loss_name, HardMarginLoss)()


    # Build a tag that distinguishes this run in the CSV
    tag = f"{FRAMEWORK}_norm={norm}_opt={opt_name}_loss={loss_name}_muon={use_muon_activations}"
    if td_lambda != 0.0:
        tag += f"_td={td_mode}{td_lambda}"
        if td_eps_lin is not None:
            tag += f"+eps{td_eps_lin}"
        if td_anneal:
            tag += "+anneal"
        if td_deflate:
            tag += "+defl"
    if num_iters != 1:
        tag += f"_it{num_iters}"

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
                yv = torch.tensor(yv, dtype=torch.long,  device=device)
                logits = model.forward(xv)
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
            # ── Fetch a new batch ─────────────────────────────────────────
            x_np, y_np = next(train_iter)
            if td_anneal and td_lambda != 0.0:
                # Linear decay of the trace strength to 0 over the run: strong
                # early-layer signal while features form, clean local
                # projections for late fine-tuning.
                ptorch_config.config.update(
                    "td_lambda", float(td_lambda) * max(0.0, 1.0 - step / cfg["max_steps"]))
            x_batch = torch.tensor(x_np, dtype=torch.float32, device=device)
            y_batch = torch.tensor(y_np, dtype=torch.long, device=device)
            y_oh    = F.one_hot(y_batch, num_classes=task_cfg["classes"]).float()

            if step >= cfg["max_steps"]:
                break


            # Initial real forward pass for this batch
            
            output = model(x_batch)
            loss(output, y_oh).sum().backward()
            optimizer.step()
            optimizer.zero_grad()

            # ── Eval ──────────────────────────────────────────────────────
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
    parser = argparse.ArgumentParser(description="PTorch MLP benchmark")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"),
                        help="Path to YAML config file")
    parser.add_argument("--max-steps", type=int, default=None, help="Override cfg['max_steps']")
    parser.add_argument("--num-runs", type=int, default=None, help="Override cfg['num_runs']")
    parser.add_argument("--td-lambda", type=float, default=0.0,
                        help="TD(λ) depth-trace strength (0 = off, exact legacy behavior)")
    parser.add_argument("--td-mode", type=str, default="norm", choices=["norm", "vector"],
                        help="trace variant (norm = seed-coupled scalar floor)")
    parser.add_argument("--muon", type=int, default=None, choices=[0, 1],
                        help="Override the cfg use_muon_activations sweep with a single value")
    parser.add_argument("--td-eps-lin", type=float, default=None,
                        help="Override config.td_eps_lin (linear-transport cap)")
    parser.add_argument("--td-anneal", action="store_true",
                        help="Linearly decay td_lambda to 0 over max_steps")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated task names to run (default: all in cfg)")
    parser.add_argument("--td-deflate", action="store_true",
                        help="Deflate the activation-parallel artifact from the trace residual")
    parser.add_argument("--num-iters", type=int, default=1,
                        help="Newton steps per bilinear solve (Linear num_iters)")
    args = parser.parse_args()

    cfg    = yaml.safe_load(open(args.config))
    if args.max_steps is not None: cfg["max_steps"] = args.max_steps
    if args.num_runs is not None: cfg["num_runs"] = args.num_runs
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Sweep axes from config (default to single values when absent)
    norms = cfg.get("norms", ["l2"])
    losses = cfg.get("losses", [cfg.get("ptorch_loss", "HardMarginLoss")])

    # Optimizer sweep: list of {name, kwargs} dicts, or fall back to the
    # single ptorch_optimizer / ptorch_optimizer_kwargs pair.
    optimizers = cfg.get("optimizers", [])
    if not optimizers:
        optimizers = [{"name": cfg.get("ptorch_optimizer", "ProjectionSGD"), 
                       "kwargs": cfg.get("kwargs", {})}]
    muon_sweeps = cfg.get("use_muon_activations", [False])
    if not isinstance(muon_sweeps, list):
        muon_sweeps = [muon_sweeps]
    if args.muon is not None:
        muon_sweeps = [bool(args.muon)]

    if args.tasks is not None:
        wanted = set(args.tasks.split(","))
        cfg["tasks"] = [t for t in cfg["tasks"] if t["name"] in wanted]

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
                                    norm=norm, opt_name=opt_name, opt_kwargs=opt_kwargs, loss_name=loss_name, use_muon_activations=muon_act,
                                    td_lambda=args.td_lambda, td_mode=args.td_mode,
                                    td_eps_lin=args.td_eps_lin, td_anneal=args.td_anneal,
                                    td_deflate=args.td_deflate, num_iters=args.num_iters)
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
