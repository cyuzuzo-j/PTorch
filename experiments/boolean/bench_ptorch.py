##################################################
###   Benchmark — ptorch                      ###
##################################################
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))

# Keep this benchmark in eager mode to avoid inductor cache/JIT crashes.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import yaml
import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.fx
from ptorch.nn.modules import Linear, LinearFrozen, ReLU, LeakyReLU, ProjectionModule, CrossEntropy, HardMarginLoss
from ptorch import config
import ptorch.optim_static as ptorch_optim_static
import tqdm, time
import wandb

FRAMEWORK = "ptorch_cyclic"
XOR_BITS = 2

OPTIM_MODULES = vars(ptorch_optim_static)
CFG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

# ── XOR Dataset ──────────────────────────────────
def make_xor_truth_table(num_bits):
    n = 1 << num_bits
    rows = torch.arange(n, dtype=torch.long)
    bits = ((rows.unsqueeze(1) >> torch.arange(num_bits - 1, -1, -1)) & 1)
    labels = (bits.sum(dim=1) % 2).long()
    return bits.float(), labels


X_XOR, Y_XOR = make_xor_truth_table(XOR_BITS)

class ProjectionTracer(torch.fx.Tracer):
    def is_leaf_module(self, m: tnn.Module, module_qualified_name: str) -> bool:
        if isinstance(m, (ProjectionModule, LeakyReLU)):
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
            print(f"set node {n} to {cache_to_apply.data}")
            result.data.copy_(cache_to_apply.data) 
        return result


# ── Model ────────────────────────────────────────
class MLP(tnn.Module):
    def __init__(self, hidden, in_features, classes):
        super().__init__()
        last = in_features
        self.hidden_layers = tnn.ModuleList()
        for i, f in enumerate(hidden):
            # Using bias=False to match JAX implementation
            if i == 0:
                self.hidden_layers.append(LinearFrozen(last, f, bias=False, g=1.0))
            else:
                self.hidden_layers.append(Linear(last, f, bias=False, g=1.0, alpha=1.0, num_iters=10))
            self.hidden_layers.append(LeakyReLU(0.1))
            last = f
        self.n_hidden = len(hidden)
        # Using bias=False and 1 output class to match JAX implementation
        self.out = Linear(last, 1, bias=False, g=1.0, alpha=1.0, num_iters=10)
        self.loss = HardMarginLoss()

    def forward(self, x, y):
        for i in range(0,len(self.hidden_layers),2):
            x = self.hidden_layers[i](x) 
            x = self.hidden_layers[i+1](x)
        x = self.out(x)
        # JAX target shape is (batch_size, 1)
        return self.loss(x, y.view(-1, 1).float())

    def forward_eval(self, x):
        for i in range(0,len(self.hidden_layers),2):
            x = self.hidden_layers[i](x) 
            x = self.hidden_layers[i+1](x)
        x = self.out(x)
        return x



# ── Training ─────────────────────────────────────
def run(cfg, task_cfg, batch_size, run_number, device):
    seed = cfg["random_seed"]
    torch.manual_seed(seed + run_number)

    model      = MLP(task_cfg["hidden"], X_XOR.shape[1], task_cfg["classes"]).to(device)

    opt_name   = cfg["ptorch_optimizer"]
    opt_kwargs = cfg.get("ptorch_optimizer_kwargs", {})
    optimizer  = OPTIM_MODULES[opt_name](model.parameters(), **opt_kwargs)

    run_name = f"{cfg.get('experiment_name', 'run')}_{FRAMEWORK}_{task_cfg['name']}_bs{batch_size}_run{run_number}_{opt_name}"
    run = wandb.init(project="pjax", name=run_name)
    wandb.config.update({
        "framework": FRAMEWORK,
        "task": task_cfg["name"],
        "architecture": str(model),
        "hidden": task_cfg["hidden"],
        "optimizer": opt_name,
        **{f"opt_{k}": v for k, v in opt_kwargs.items()},
        "batch_size": batch_size,
        "seed": seed,
        "run_number": run_number,
        "max_steps": cfg["max_steps"],
        "eval_every": cfg["eval_every"],
        "patience": cfg["patience"],
    })

    def eval_fn(x, y):
        with torch.no_grad():
            logits = model.forward_eval(x)
            preds = (logits > 0.5).long().squeeze()
            return (preds == y).float().mean()

    best_val_acc, best_state, best_step = 0.0, None, 0
    no_improve = 0
    step = 0
    t0 = time.time()

    x_train, y_train = X_XOR.to(device), Y_XOR.to(device)

    ## ------ initial forward, then backward+propagate each step --------
    tracer = ProjectionTracer()
    graph = tracer.trace(model)   
    traced = torch.fx.GraphModule(model, graph)

    traced.train()
    output = traced(x_train, y_train)

    def step_fn():
        nonlocal output
        optimizer.zero_grad()
        # Projection losses return logits (non-scalar), so seed backward explicitly.
        print("--------------------------")
        output.sum().backward()
        optimizer.step()
        print("-------------------------------")

        # Keep a scalar metric for logging that matches the hard-margin objective.
        with torch.no_grad():
            logits = model.forward_eval(x_train)
            target = y_train.view(-1, 1).float()
            err1 = torch.where(target == 1, torch.where(logits < 1.0, (logits - 1.0) ** 2, torch.tensor(0.0, device=logits.device)), torch.tensor(0.0, device=logits.device))
            err0 = torch.where(target == 0, torch.where(logits > 0.0, logits ** 2, torch.tensor(0.0, device=logits.device)), torch.tensor(0.0, device=logits.device))
            scalar_loss = (err1 + err0).mean()

        output = PropagateCache(traced, skip_modules={'out'}).run(x_train, y_train)
        return float(scalar_loss.item())


    with tqdm.tqdm(unit="step") as pbar:
        while True:
            if step % cfg["eval_every"] == 0:
                model.eval()
                val_acc = float(eval_fn(x_train, y_train))
                model.train()
                wandb.log({"val/val_acc": val_acc}, step=step)
                wandb.log({"training_time_s": time.time() - t0}, step=step)
                pbar.set_postfix(val_acc=f"{val_acc:.4f}", best=f"{best_val_acc:.4f}")
                if val_acc >= best_val_acc:
                    best_val_acc, best_step = val_acc, step
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= cfg["patience"]:
                    print(f"Early stop at step {step}")
                    break

            loss = step_fn()
            wandb.log({"train/loss": float(loss)}, step=step)
            step += 1
            pbar.update(1)
            if cfg["max_steps"] and step >= cfg["max_steps"]:
                break

    total_time = time.time() - t0
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    final_acc = float(eval_fn(x_train, y_train))
    print(f"Test Acc: {final_acc:.4f}  Time: {total_time:.1f}s")
    wandb.log({"test/test_acc": final_acc}, step=step)
    wandb.log({"total_training_time_s": total_time}, step=step)
    wandb.finish()
    return final_acc


if __name__ == "__main__":
    cfg    = yaml.safe_load(open(CFG_PATH))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for batch_size in cfg["batch_sizes"]:
        for task_cfg in cfg["tasks"]:
            print(f"\n{'='*50}\n{FRAMEWORK} | {task_cfg['name']} | bs={batch_size}")
            for run_number in range(1, cfg["num_runs"] + 1):
                run(cfg, task_cfg, batch_size, run_number, device)
