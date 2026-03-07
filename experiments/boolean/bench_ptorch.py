##################################################
###   Benchmark — ptorch                      ###
##################################################
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))

import yaml
import torch
import torch.nn as tnn
import torch.nn.functional as F
from ptorch.nn.modules import LinearBias, Simplex, ReLU, MultiHeadAttention
from ptorch.core.ops import CrossEntropyProjection
import ptorch.optim_static as ptorch_optim_static
import ptorch.config as ptorch_config
import tqdm, time
import wandb

FRAMEWORK = "ptorch_simplex"

OPTIM_MODULES = vars(ptorch_optim_static)
CFG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

# ── XOR Dataset ──────────────────────────────────
X_XOR = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y_XOR = torch.tensor([0, 1, 1, 0], dtype=torch.long)

# ── Model ────────────────────────────────────────
class MLP(tnn.Module):
    def __init__(self, hidden, in_features, classes):
        super().__init__()
        last = in_features
        self.hidden_layers = tnn.ModuleList()
        for f in hidden:
            # Using same g and alpha from CNN/MLP
            self.hidden_layers.append(LinearBias(last, f, g=0.885, alpha=120.11))
            self.hidden_layers.append(ReLU(f))
            last = f
        self.n_hidden = len(hidden)
        self.out = LinearBias(last, classes)

    def forward(self, x):
        for i in range(0,len(self.hidden_layers),2):
            x = self.hidden_layers[i](x) 
            x = self.hidden_layers[i+1](x)
        return self.out(x)


# ── Training ─────────────────────────────────────
def run(cfg, task_cfg, batch_size, run_number, device):
    seed = cfg["random_seed"]
    torch.manual_seed(seed + run_number)

    model      = MLP(task_cfg["hidden"], task_cfg["in_features"], task_cfg["classes"]).to(device)
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
        **ptorch_config.snapshot(),
    })

    def step_fn(x, y):
        logits  = model(x)
        y_oh    = F.one_hot(y, num_classes=logits.shape[-1]).float()
        projected = CrossEntropyProjection.apply(logits, y_oh)
        optimizer.zero_grad()
        projected.sum().backward()
        loss = F.cross_entropy(logits.detach(), y)
        optimizer.step()
        return loss

    def eval_fn(x, y):
        with torch.no_grad():
            return (model(x).argmax(dim=-1) == y).float().mean()

    best_val_acc, best_state, best_step = 0.0, None, 0
    no_improve = 0
    step = 0
    t0 = time.time()

    x_train, y_train = X_XOR.to(device), Y_XOR.to(device)

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

            loss = step_fn(x_train, y_train)
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
