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
from ptorch.nn.modules import Linear, LinearFrozen, LeakyReLU
import ptorch.optim_static as ptorch_optim_static
from ptorch.core.ops import HardMarginProjection
import tqdm, time
import wandb

FRAMEWORK = "ptorch_normal_normal"
XOR_BITS = 8
XOR_TRAIN_SAMPLES = 256

OPTIM_MODULES = vars(ptorch_optim_static)
CFG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

# ── XOR Dataset ──────────────────────────────────
def make_xor_dataset(num_bits, n_samples, seed, device):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    x = torch.randint(0, 2, (n_samples, num_bits), generator=g, dtype=torch.long)
    y = (x.sum(dim=1) % 2).long()
    return x.float().to(device), y.to(device)

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

    def forward(self, x):
        for i in range(0,len(self.hidden_layers),2):
            x = self.hidden_layers[i](x) 
            x = self.hidden_layers[i+1](x)
        return self.out(x)


# ── Training ─────────────────────────────────────
def run(cfg, task_cfg, batch_size, run_number, device):
    seed = cfg["random_seed"]
    torch.manual_seed(seed + run_number)

    in_features = XOR_BITS

    model      = MLP(task_cfg["hidden"], in_features, task_cfg["classes"]).to(device)
    opt_name   = cfg["ptorch_optimizer"]
    opt_kwargs = cfg.get("ptorch_optimizer_kwargs", {})
    optimizer  = OPTIM_MODULES[opt_name](model.parameters(), **opt_kwargs)

    run_name = f"{cfg.get('experiment_name', 'run')}_{FRAMEWORK}_{task_cfg['name']}_bs{batch_size}_run{run_number}_{opt_name}"
    run = wandb.init(project="pjax", name=run_name)
    wandb.config.update({
        "framework": FRAMEWORK,
        "task": task_cfg["name"],
        "xor_bits": in_features,
        "xor_train_samples": XOR_TRAIN_SAMPLES,
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

    def step_fn(x, y):
        optimizer.zero_grad()
        logits = model(x)
        projection = HardMarginProjection.apply(logits, y.view(-1, 1).float())
        # HardMarginProjection returns a tensor; seed backward with an explicit scalar.
        projection.sum().backward()
        optimizer.step()
        
        return projection.mean().detach()

    def eval_fn(x, y):
        with torch.no_grad():
            logits = model(x)
            preds = (logits > 0.5).long().squeeze()
            return (preds == y).float().mean()

    best_val_acc, best_state, best_step = 0.0, None, 0
    no_improve = 0
    step = 0
    t0 = time.time()

    x_train, y_train = make_xor_dataset(
        num_bits=in_features,
        n_samples=XOR_TRAIN_SAMPLES,
        seed=seed + run_number,
        device=device,
    )

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
