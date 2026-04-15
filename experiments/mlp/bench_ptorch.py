##################################################
###   Benchmark — ptorch                      ###
###   Alternating projections (PyTorch)       ###
##################################################
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))

import gc
import yaml
import torch
import torch.nn as tnn
import torch.nn.functional as F
from ptorch.nn.modules import LinearHybrid, ReLUHybrid, Linear as PLinear, ReLU as PReLU, LinearAbs, _parse_norm
from ptorch.core.ops import CrossEntropyProjection, HardMarginProjection, ProximalHingeMargin, SmoothSoftMargin
import ptorch.optim_static as ptorch_optim_static
import ptorch.config as ptorch_config
from experiments.shared.data import MNISTDataModule, InfiniteCifarDataModule
import tqdm, time
import wandb

FRAMEWORK = "ptorch_hybrid"

OPTIM_MODULES = vars(ptorch_optim_static)
CFG_PATH = "run_sgd.yaml" if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "config.yaml")

DATASETS = {"MNIST": MNISTDataModule, "CIFAR10": InfiniteCifarDataModule}

# ── Loss projection registry ──────────────────────
# Maps config key → autograd.Function class.
# All classes must accept (logits, one_hot_labels) in .apply().
LOSS_PROJECTIONS = {
    "CrossEntropyProjection": CrossEntropyProjection,
    "HardMarginProjection":   HardMarginProjection,
    "ProximalHingeMargin":    ProximalHingeMargin,
    "SmoothSoftMargin":       SmoothSoftMargin,
}


# ── Model ────────────────────────────────────────
class MLP(tnn.Module):
    def __init__(self, hidden, in_features, classes, norm="l2", dtp=False, use_hybrid=False, use_fused_relu=False):
        super().__init__()
        last = in_features
        self.hidden_layers = tnn.ModuleList()
        self.use_fused_relu = use_fused_relu
        for f in hidden:
            if use_fused_relu:
                # Fused Linear+ReLU using exact ℓ∞ projection
                self.hidden_layers.append(LinearAbs(last, f, ))
            elif use_hybrid:
                self.hidden_layers.append(LinearHybrid(last, f, norm=norm))
                self.hidden_layers.append(ReLUHybrid(norm=norm))
            else:
                self.hidden_layers.append(PLinear(last, f, norm=norm, dtp=dtp))
                self.hidden_layers.append(PReLU(norm=norm if norm in ('l2', 'linf') else 'l2'))
            last = f
        self.n_hidden = len(hidden)
        if use_hybrid:
            self.out = LinearHybrid(last, classes, norm=norm)
        else:
            self.out = LinearAbs(last, classes)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        if self.use_fused_relu:
            for layer in self.hidden_layers:
                x = layer(x)
        else:
            for i in range(0, len(self.hidden_layers), 2):
                x = self.hidden_layers[i](x)
                x = self.hidden_layers[i + 1](x)
        return self.out(x)


# ── Training ─────────────────────────────────────
def run(cfg, task_cfg, batch_size, run_number, device, loss_projection_cls=CrossEntropyProjection, log_loss_projection=False):
    seed = cfg["random_seed"]
    run_seed = seed + run_number
    torch.manual_seed(run_seed)
    norm = cfg.get("ptorch_norm", 2)
    dtp = cfg.get("ptorch_dtp", False)
    use_hybrid = cfg.get("ptorch_use_hybrid", False)
    use_fused_relu = cfg.get("ptorch_use_fused_relu", False)

    norm_type, p_val = _parse_norm(norm)
    model_norm = norm_type
    FRAMEWORK = "ptorch_hybrid" if use_hybrid else "ptorch"
    
    # Set global config for the projection norm
    if norm_type == 'lp':
        ptorch_config.update("projection_norm", "lp")
        ptorch_config.update("projection_p", p_val)
    else:
        ptorch_config.update("projection_norm", model_norm)
        ptorch_config.update("projection_p", p_val)
    dataset_cls = DATASETS[task_cfg["name"]]
    ds = dataset_cls(batch_size=batch_size, seed=run_seed)
    train_iter = ds.train_iterator()
    val_loader  = ds.val_dataloader()
    test_loader = ds.test_dataloader()

    model      = MLP(task_cfg["hidden"], task_cfg["in_features"], task_cfg["classes"], norm=norm, dtp=dtp, use_hybrid=use_hybrid, use_fused_relu=use_fused_relu).to(device)
    opt_name   = cfg["ptorch_optimizer"]
    opt_kwargs = cfg.get("ptorch_optimizer_kwargs", {})
    optimizer  = OPTIM_MODULES[opt_name](model.parameters(), **opt_kwargs)

    loss_proj_name = loss_projection_cls.__name__
    run_name = f"{cfg.get('experiment_name', 'run')}_{FRAMEWORK}_{task_cfg['name']}_bs{batch_size}_run{run_number}_{opt_name}_{loss_proj_name}"
    run = wandb.init(project="pjax", name=run_name)
    wandb.config.update({
        "framework": FRAMEWORK,
        "task": task_cfg["name"],
        "hidden": task_cfg["hidden"],
        "optimizer": opt_name,
        **{f"opt_{k}": v for k, v in opt_kwargs.items()},
        "batch_size": batch_size,
        "seed": run_seed,
        "base_seed": seed,
        "run_number": run_number,
        "max_steps": cfg["max_steps"],
        "eval_every": cfg["eval_every"],
        "patience": cfg["patience"],
        "loss_projection": loss_proj_name,
        "log_loss_projection": log_loss_projection,
        "ptorch_norm": norm,
        "ptorch_norm_type": norm_type,
        "ptorch_p": p_val,
        "ptorch_dtp": dtp,
        "ptorch_use_hybrid": use_hybrid,
        **ptorch_config.snapshot(),
    })

    def step_fn(x, y):
        logits  = model(x)
        y_oh    = F.one_hot(y.long(), num_classes=logits.shape[-1]).float()
        optimizer.zero_grad()
        if "hybrid" not in FRAMEWORK:
            # Pure projection mode: loss layer returns an absolute TARGET
            projected = loss_projection_cls.apply(logits, y_oh)
            proj_loss = projected.sum()
            proj_loss.backward()
            loss = F.cross_entropy(logits.detach(), y.long())
            proj_loss_val = float(proj_loss.detach())
        else:
            # Hybrid or regular gradient mode: standard backward gradients
            loss = F.cross_entropy(logits, y.long())
            loss.backward()
            proj_loss_val = 0.0
        optimizer.step()
        return loss, proj_loss_val

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

            x, y = next(train_iter)
            loss, proj_loss_val = step_fn(
                torch.tensor(x, dtype=torch.float32, device=device),
                torch.tensor(y, dtype=torch.long,  device=device))
            log_dict = {"train/loss": float(loss)}
            if log_loss_projection:
                log_dict["train/loss_projection"] = proj_loss_val
            wandb.log(log_dict, step=step)
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
    wandb.log({"test/test_acc": final_acc}, step=step)
    wandb.log({"total_training_time_s": total_time}, step=step)
    wandb.finish()

if __name__ == "__main__":
    cfg    = yaml.safe_load(open(CFG_PATH))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for batch_size in cfg["batch_sizes"]:
        norm_sweep = cfg.get("ptorch_norm_sweep", [cfg.get("ptorch_norm", 2)])
        for norm_val in norm_sweep:
            for task_cfg in cfg["tasks"]:
                norm_type, p_val = _parse_norm(norm_val)
                norm_label = f"l{p_val}" if norm_type == 'lp' else norm_type
                use_hybrid = cfg.get("ptorch_use_hybrid", True)
                FRAMEWORK = "ptorch_hybrid" if use_hybrid else "ptorch"
                print(f"\n{'='*50}\n{FRAMEWORK} | {task_cfg['name']} | bs={batch_size} | norm={norm_label}")
                # Override the norm for this sweep iteration
                sweep_cfg = dict(cfg)
                sweep_cfg["ptorch_norm"] = norm_val
                for run_number in range(1, cfg["num_runs"] + 1):
                    proj_name = cfg.get("ptorch_loss_projection", "CrossEntropyProjection")
                    proj_cls  = LOSS_PROJECTIONS[proj_name]
                    run(sweep_cfg, task_cfg, batch_size, run_number, device,
                        loss_projection_cls=proj_cls,
                        log_loss_projection=cfg.get("log_loss_projection", False))
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
