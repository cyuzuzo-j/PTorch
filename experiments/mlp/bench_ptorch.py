import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))

import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.fx
from ptorch.nn.modules import Linear as PLinear, ReLU as PReLU, ProjectionModule, CrossEntropy
from ptorch import config as ptorch_config
import ptorch.optim_static as ptorch_optim_static

ptorch_config.use_projections = True

import yaml
import wandb
import tqdm, time
from experiments.shared.data import MNISTDataModule

CFG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
FRAMEWORK = "ptorch_CyclicProjections_ManyBackward"


# ── FX Tracer/Interpreter (same as boolean example) ──────────────────────────

class ProjectionTracer(torch.fx.Tracer):
    def is_leaf_module(self, m: tnn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, ProjectionModule):
            return True
        return super().is_leaf_module(m, module_qualified_name)


class PropagateCache(torch.fx.Interpreter):
    def run_node(self, n: torch.fx.Node):
        cache_to_apply = None
        if n.op == 'call_module':
            submod = self.module.get_submodule(n.target)
            if hasattr(submod, 'projection_forward_cache') and isinstance(submod.projection_forward_cache, list):
                if submod.projection_forward_cache:
                    cache = submod.projection_forward_cache[0]
                    if cache is not None:
                        cache_to_apply = cache
                    else:
                        print(f"\n[PropagateCache] Intercepted node '{n.name}' but cache is empty.")

        result = super().run_node(n)

        if cache_to_apply is not None:
            result.data.copy_(cache_to_apply.data)

        return result


# ── Model ─────────────────────────────────────────────────────────────────────

class MLP(tnn.Module):
    def __init__(self, hidden, in_features, classes):
        super().__init__()
        last = in_features
        self.hidden_layers = tnn.ModuleList()
        for f in hidden:
            self.hidden_layers.append(PLinear(last, f))
            self.hidden_layers.append(PReLU())
            last = f
        self.out = PLinear(last, classes, g=float('inf'))
        self.cross_entropy = CrossEntropy()

    def get_logits(self, x):
        x = x.reshape(x.shape[0], -1)
        for i in range(0, len(self.hidden_layers), 2):
            x = self.hidden_layers[i](x)
            x = self.hidden_layers[i + 1](x)
        return self.out(x)

    def forward(self, x, y_oh):
        return self.cross_entropy(self.get_logits(x), y_oh)


# ── Training ──────────────────────────────────────────────────────────────────

def run(cfg, task_cfg, batch_size, run_number, device):
    seed = cfg["random_seed"]
    run_seed = seed + run_number
    torch.manual_seed(run_seed)

    ds = MNISTDataModule(batch_size=batch_size, seed=run_seed)
    train_iter = ds.train_iterator()
    val_loader  = ds.val_dataloader()
    test_loader = ds.test_dataloader()

    # Fix a single training batch for the many-backward loop
    x_np, y_np = next(train_iter)
    x_batch = torch.tensor(x_np, dtype=torch.float32, device=device)
    y_batch = torch.tensor(y_np, dtype=torch.long, device=device)
    y_oh    = F.one_hot(y_batch, num_classes=task_cfg["classes"]).float()

    model = MLP(task_cfg["hidden"], task_cfg["in_features"], task_cfg["classes"]).to(device)

    # Trace the model once
    tracer = ProjectionTracer()
    graph  = tracer.trace(model)
    traced = torch.fx.GraphModule(model, graph)

    opt_name   = cfg["ptorch_optimizer"]
    opt_kwargs = cfg.get("ptorch_optimizer_kwargs", {})
    OPTIM_MODULES = vars(ptorch_optim_static)
    optimizer  = OPTIM_MODULES[opt_name](traced.parameters(), **opt_kwargs)

    run_name = (
        f"{cfg.get('experiment_name', 'run')}_{FRAMEWORK}"
        f"_{task_cfg['name']}_bs{batch_size}_run{run_number}_{opt_name}"
    )

    run_wandb = wandb.init(project="pjax", name=run_name)
    wandb.config.update({
        "framework": FRAMEWORK,
        "task": task_cfg["name"],
        "hidden": task_cfg["hidden"],
        "optimizer": opt_name,
        **{f"opt_{k}": v for k, v in opt_kwargs.items()},
        "batch_size": batch_size,
        "seed": run_seed,
        "run_number": run_number,
        "max_steps": cfg["max_steps"],
        "eval_every": cfg["eval_every"],
        "patience": cfg["patience"],
    })

    def eval_acc(loader):
        traced.eval()
        accs = []
        with torch.no_grad():
            for xv, yv in loader:
                xv = torch.tensor(xv, dtype=torch.float32, device=device)
                yv = torch.tensor(yv, dtype=torch.long,  device=device)
                yv_oh = F.one_hot(yv, num_classes=task_cfg["classes"]).float()
                logits = traced(xv, yv_oh)
                accs.append((logits.argmax(dim=-1) == yv).float().mean())
        traced.train()
        return float(torch.stack(accs).mean())

    # ── Initial forward pass (same pattern as boolean example) ────────────────
    traced.train()
    output = traced(x_batch, y_oh)

    best_val_acc, best_state, best_step = 0.0, None, 0
    no_improve = 0
    t0 = time.time()

    with tqdm.tqdm(total=cfg["max_steps"], unit="step") as pbar:
        for step in range(cfg["max_steps"]):

            # ── Eval ──────────────────────────────────────────────────────────
            if step % cfg["eval_every"] == 0:
                val_acc = eval_acc(val_loader)
                wandb.log({"val/val_acc": val_acc, "training_time_s": time.time() - t0}, step=step)
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

            # ── Backward + weight update ───────────────────────────────────
            optimizer.zero_grad()
            output.sum().backward()
            optimizer.step()

            with torch.no_grad():
                logits = model.get_logits(x_batch)
            ce_loss   = float(F.cross_entropy(logits, y_batch))
            train_acc = float((logits.argmax(dim=-1) == y_batch).float().mean())
            wandb.log({"train/loss": ce_loss, "train/train_acc": train_acc}, step=step)

            # ── Propagate projections (no new forward pass) ───────────────
            interpreter = PropagateCache(traced)
            output = interpreter.run(x_batch, y_oh)

            pbar.update(1)

    total_time = time.time() - t0
    if best_state:
        traced.load_state_dict(best_state)

    test_acc = eval_acc(test_loader)
    print(f"Test Acc: {test_acc:.4f}  Time: {total_time:.1f}s")
    wandb.log({"test/test_acc": test_acc, "total_training_time_s": total_time}, step=step)
    wandb.finish()


if __name__ == "__main__":
    cfg    = yaml.safe_load(open(CFG_PATH))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for batch_size in cfg["batch_sizes"]:
        for task_cfg in cfg["tasks"]:
            print(f"\n{'='*50}\n{FRAMEWORK} | {task_cfg['name']} | bs={batch_size}")
            for run_number in range(1, cfg["num_runs"] + 1):
                run(cfg, task_cfg, batch_size, run_number, device)