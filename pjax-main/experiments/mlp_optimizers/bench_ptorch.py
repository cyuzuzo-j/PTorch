##################################################
###   Benchmark — ptorch                      ###
###   Alternating projections (PyTorch)       ###
##################################################
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import gc
import yaml
import torch
import torch.nn as tnn
import torch.nn.functional as F
from ptorch.nn.modules import LinearBias, ReLU
from ptorch.core.ops import MarginLossProjection
import ptorch.optim_static as ptorch_optim_static
from experiments.shared.data import MNISTDataModule, InfiniteCifarDataModule
import tqdm, time
from aim import Run

FRAMEWORK = "ptorch_compiled_itterative"

OPTIM_MODULES = vars(ptorch_optim_static)
CFG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

DATASETS = {"MNIST": MNISTDataModule, "CIFAR10": InfiniteCifarDataModule}


# ── Model ────────────────────────────────────────
class MLP(tnn.Module):
    def __init__(self, hidden, in_features, classes):
        super().__init__()
        last = in_features
        self.hidden_layers = tnn.ModuleList()
        for f in hidden:
            self.hidden_layers.append(LinearBias(last, f))
            self.hidden_layers.append(ReLU(f))
            last = f
        self.n_hidden = len(hidden)
        self.out = LinearBias(last, classes)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        for i in range(0, len(self.hidden_layers), 2):
            x = self.hidden_layers[i](x)      # LinearBias
            x = self.hidden_layers[i + 1](x)  # ReLU
        return self.out(x)


# ── Training ─────────────────────────────────────
def run(cfg, task_cfg, batch_size, run_number, device):
    seed = cfg["random_seed"]
    torch.manual_seed(seed + run_number)

    dataset_cls = DATASETS[task_cfg["name"]]
    ds = dataset_cls(batch_size=batch_size, seed=seed)
    train_iter = ds.train_iterator()
    val_loader  = ds.val_dataloader()
    test_loader = ds.test_dataloader()

    model      = MLP(task_cfg["hidden"], task_cfg["in_features"], task_cfg["classes"]).to(device)
    opt_name   = cfg["ptorch_optimizer"]
    opt_kwargs = cfg.get("ptorch_optimizer_kwargs", {})
    optimizer  = OPTIM_MODULES[opt_name](model.parameters(), **opt_kwargs)

    run = Run(experiment=cfg["experiment_name"])
    run["hparams"] = {
        "framework": FRAMEWORK,
        "task": task_cfg["name"],
        "hidden": task_cfg["hidden"],
        "optimizer": opt_name,
        **{f"opt_{k}": v for k, v in opt_kwargs.items()},
        "batch_size": batch_size,
        "seed": seed,
        "run_number": run_number,
        "max_steps": cfg["max_steps"],
        "eval_every": cfg["eval_every"],
        "patience": cfg["patience"],
    }

    def step_fn(x, y):
        logits  = model(x)
        y_oh    = F.one_hot(y.long(), num_classes=logits.shape[-1]).float()
        projected = MarginLossProjection.apply(logits, y_oh)
        optimizer.zero_grad()
        projected.sum().backward()
        loss = F.cross_entropy(logits.detach(), y.long())
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
                    torch.tensor(x, dtype=torch.float32, device=device),
                    torch.tensor(y, dtype=torch.long,  device=device))
                    for x, y in val_loader]
                val_acc = float(torch.stack(accs).mean())
                model.train()
                run.track(val_acc, name="val_acc", step=step, context={"subset": "val"})
                run.track(time.time() - t0, name="training_time_s", step=step)
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
            loss = step_fn(
                torch.tensor(x, dtype=torch.float32, device=device),
                torch.tensor(y, dtype=torch.long,  device=device))
            run.track(float(loss), name="loss", step=step, context={"subset": "train"})
            step += 1
            pbar.update(1)
            if cfg["max_steps"] and step >= cfg["max_steps"]:
                break

    total_time = time.time() - t0
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    test_accs = [eval_fn(
        torch.tensor(x, dtype=torch.float32, device=device),
        torch.tensor(y, dtype=torch.long,  device=device))
        for x, y in test_loader]
    final_acc = float(torch.stack(test_accs).mean())
    print(f"Test Acc: {final_acc:.4f}  Time: {total_time:.1f}s")
    run.track(final_acc, name="test_acc", step=step, context={"subset": "test"})
    run.track(total_time, name="total_training_time_s", step=step)
    run.close()
    return final_acc, best_val_acc, best_step, total_time


if __name__ == "__main__":
    cfg    = yaml.safe_load(open(CFG_PATH))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for batch_size in cfg["batch_sizes"]:
        for task_cfg in cfg["tasks"]:
            print(f"\n{'='*50}\n{FRAMEWORK} | {task_cfg['name']} | bs={batch_size}")
            for run_number in range(1, cfg["num_runs"] + 1):
                run(cfg, task_cfg, batch_size, run_number, device)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
