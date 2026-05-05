##################################################
###   Benchmark — ptorch (cyclic projections)  ###
###   Supports sweeps over norms & optimizers  ###
##################################################
import sys, os, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))
import gc
import yaml
import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.fx
import pandas as pd
from ptorch.nn.modules import (
    Linear, LinearFrozen, LeakyReLU, ReLU,
    ProjectionModule, CrossEntropy, HardMarginLoss, ProximalHingeMarginLoss
)
import ptorch.nn.modules as ptorch_modules
from ptorch import config as ptorch_config
import ptorch.optim_static as ptorch_optim_static
from experiments.shared.data import MNISTDataModule, InfiniteCifarDataModule
import tqdm, time

ptorch_config.use_projections = True

FRAMEWORK = "ptorch"
OPTIM_MODULES = vars(ptorch_optim_static)

DATASETS = {"MNIST": MNISTDataModule, "CIFAR10": InfiniteCifarDataModule}


# ── FX Tracer / Interpreter ─────────────────────────────────────────────────

class ProjectionTracer(torch.fx.Tracer):
    def is_leaf_module(self, m: tnn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, (ProjectionModule, ReLU, LeakyReLU)):
            return True
        return super().is_leaf_module(m, module_qualified_name)


class PropagateCache(torch.fx.Interpreter):
    def __init__(self, module, skip_modules=None):
        super().__init__(module)
        self.skip_targets = skip_modules or set()

    def run_node(self, n: torch.fx.Node):
        cache_to_apply = None
        if n.op == 'call_module' and n.target not in self.skip_targets:
            submod = self.module.get_submodule(n.target)
            if hasattr(submod, 'projection_forward_cache') and isinstance(submod.projection_forward_cache, list):
                if submod.projection_forward_cache:
                    cache = submod.projection_forward_cache[0]
                    if cache is not None:
                        cache_to_apply = cache

        result = super().run_node(n)

        if cache_to_apply is not None:
            result.data.copy_(cache_to_apply.data)

        return result


# ── Model ────────────────────────────────────────────────────────────────────

class MLP(tnn.Module):
    def __init__(self, hidden, in_features, classes, norm='l2', loss_name='HardMarginLoss'):
        super().__init__()
        last = in_features
        self.hidden_layers = tnn.ModuleList()
        for i, f in enumerate(hidden):
            if i == 0:
                self.hidden_layers.append(LinearFrozen(in_features, f, bias=False))
            else:
                self.hidden_layers.append(Linear(last, f, bias=False, residual=True, norm=norm))
            self.hidden_layers.append(LeakyReLU(0.1))
            last = f
        self.out = Linear(last, classes, bias=False, norm=norm)
        loss_cls = getattr(ptorch_modules, loss_name, HardMarginLoss)
        self.loss = loss_cls()

    def forward(self, x, y_oh):
        x = x.reshape(x.shape[0], -1)
        for i in range(0, len(self.hidden_layers), 2):
            x = self.hidden_layers[i](x)
            x = self.hidden_layers[i + 1](x)
        x = self.out(x)
        return self.loss(x, y_oh)

    def forward_eval(self, x):
        x = x.reshape(x.shape[0], -1)
        for i in range(0, len(self.hidden_layers), 2):
            x = self.hidden_layers[i](x)
            x = self.hidden_layers[i + 1](x)
        return self.out(x)


# ── Training ─────────────────────────────────────────────────────────────────

def run(cfg, task_cfg, batch_size, run_number, device,
        norm='l2', opt_name=None, opt_kwargs=None, loss_name='HardMarginLoss', use_muon_activations=False):
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
                task_cfg["classes"], norm=norm, loss_name=loss_name).to(device)

    ptorch_config.use_muon_activations = use_muon_activations

    # Trace the model once with ProjectionTracer
    tracer = ProjectionTracer()
    graph  = tracer.trace(model)
    traced = torch.fx.GraphModule(model, graph)

    opt_name   = opt_name   or cfg["ptorch_optimizer"]
    opt_kwargs = opt_kwargs if opt_kwargs is not None else cfg.get("ptorch_optimizer_kwargs", {})
    optimizer  = OPTIM_MODULES[opt_name](traced.parameters(), **opt_kwargs)

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
                yv = torch.tensor(yv, dtype=torch.long,  device=device)
                logits = model.forward_eval(xv)
                accs.append((logits.argmax(dim=-1) == yv).float().mean())
        model.train()
        return float(torch.stack(accs).mean())

    best_val_acc, best_state, best_step = 0.0, None, 0
    no_improve = 0
    t0 = time.time()

    K = cfg.get("K", 1)

    step = 0
    with tqdm.tqdm(total=cfg["max_steps"], unit="step",
                   desc=f"{task_cfg['name']} | norm={norm} | {opt_name} | {loss_name}") as pbar:
        while step < cfg["max_steps"]:
            # ── Fetch a new batch ─────────────────────────────────────────
            x_np, y_np = next(train_iter)
            x_batch = torch.tensor(x_np, dtype=torch.float32, device=device)
            y_batch = torch.tensor(y_np, dtype=torch.long, device=device)
            y_oh    = F.one_hot(y_batch, num_classes=task_cfg["classes"]).float()

            traced.train()
            # Initial real forward pass for this batch
            output = traced(x_batch, y_oh)

            for _ in range(K):
                if step >= cfg["max_steps"]:
                    break

                # ── Backward + weight update ──────────────────────────────
                optimizer.zero_grad()
                output.sum().backward()
                optimizer.step()

                # ── Propagate projections (no new forward pass) ───────────
                interpreter = PropagateCache(traced, skip_modules={'out'})
                output = interpreter.run(x_batch, y_oh)

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
                    best_state = {k: v.clone() for k, v in traced.state_dict().items()}
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
        traced.load_state_dict(best_state)

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
    args = parser.parse_args()

    cfg    = yaml.safe_load(open(args.config))
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Sweep axes from config (default to single values when absent)
    norms = cfg.get("norms", ["l2"])
    losses = cfg.get("losses", [cfg.get("ptorch_loss", "HardMarginLoss")])

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
