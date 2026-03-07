##################################################
###   Benchmark — torch (PyTorch baseline)    ###
##################################################
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))

import yaml
import torch
import torch.nn as tnn
import torch.nn.functional as F
import tqdm, time
import wandb

FRAMEWORK = "torch"
CFG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

# ── XOR Dataset ──────────────────────────────────
X_XOR = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y_XOR = torch.tensor([0, 1, 1, 0], dtype=torch.long)

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
        return self.out(self.body(x))


# ── Training ─────────────────────────────────────
def run(cfg, task_cfg, batch_size, run_number, device):
    seed = cfg["random_seed"]
    torch.manual_seed(seed + run_number)

    model      = MLP(task_cfg["hidden"], task_cfg["in_features"], task_cfg["classes"]).to(device)
    opt_name   = cfg["torch_optimizer"]
    opt_kwargs = cfg.get("torch_optimizer_kwargs", {})
    optimizer  = getattr(torch.optim, opt_name)(model.parameters(), **opt_kwargs)

    run = wandb.init(project="pjax", name=cfg["experiment_name"])
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

    def step_fn(x, y):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
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
                if val_acc > best_val_acc:
                    best_val_acc, best_step = val_acc, step
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= cfg["patience"]:
                    print(f"Early stop at step {step}")
                    break
                # Since XOR is small and deterministically simple, we can stop at 100% accuracy
                if val_acc == 1.0:
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
