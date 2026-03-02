##################################################
###   Benchmark — pjax (new)                  ###
###   Alternating projections via optim_static ###
##################################################
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))

import gc
import yaml
import jax
import jax.numpy as jnp
import numpy as np
import pjax
from pjax import nn, optim_static, optim
from experiments.shared.data import MNISTDataModule, InfiniteCifarDataModule
import tqdm, time
import wandb

FRAMEWORK = "pjax_blend"

OPTIM_MODULES = {**vars(optim_static), **vars(optim)}
CFG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

DATASETS = {"MNIST": MNISTDataModule, "CIFAR10": InfiniteCifarDataModule}


# ── Model ────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, hidden, in_features, classes):
        super().__init__()
        last = in_features
        for i, f in enumerate(hidden):
            setattr(self, f"dense_{i}", nn.LinearOld(last, f))
            setattr(self, f"relu_{i}",  nn.ReLU(f))
            last = f
        self.n_hidden = len(hidden)
        self.out = nn.LinearOld(last, classes)
        
    def get_params(self, key, init_x):
        return self.init(key)

    def __call__(self, x):
        x = pjax.reshape(x, (x.shape[0], -1))
        for i in range(self.n_hidden):
            x = getattr(self, f"dense_{i}")(x)
            x = getattr(self, f"relu_{i}")(x)
        return self.out(x)


# ── Training ─────────────────────────────────────
def run(cfg, task_cfg, batch_size, run_number, key):
    seed = cfg["random_seed"]
    torch_seed = seed + run_number

    dataset_cls = DATASETS[task_cfg["name"]]
    data_seed   = int(jax.random.randint(key, (), 0, 2**30))
    ds = dataset_cls(batch_size=batch_size, seed=data_seed)
    train_iter = ds.train_iterator()
    val_loader  = ds.val_dataloader()
    test_loader = ds.test_dataloader()

    model_key, _ = jax.random.split(key)
    model  = MLP(task_cfg["hidden"], task_cfg["in_features"], task_cfg["classes"])
    init_x = jnp.array(next(iter(val_loader))[0])
    params = model.get_params(model_key, init_x)
    opt_name   = cfg["pjax_optimizer"]
    opt_kwargs = cfg.get("pjax_optimizer_kwargs", {})
    optimizer  = OPTIM_MODULES[opt_name](**opt_kwargs)

    run = wandb.init(project="pjax", name=cfg["experiment_name"])
    wandb.config.update({
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
    })
    @jax.jit
    def step_fn(params, x, y):
        def apply_fn(p):
            pred = model.apply(p, x)
            y_oh = jax.nn.one_hot(y, pred.shape[-1])
            return pjax.cross_entropy(pred, y_oh)
        loss = jnp.mean(apply_fn(params))
        new_params = optimizer.update(apply_fn, params)[0]
        return new_params, loss
    
    @jax.jit
    def eval_fn(params, x, y):
        pred = model.apply(params, x)
        if hasattr(pred, 'value'):
            pred = pred.value
        return jnp.mean(jnp.argmax(pred, axis=-1) == y)

    best_val_acc, best_params, best_step = 0.0, params, 0
    no_improve = 0
    step = 0
    t0 = time.time()

    with tqdm.tqdm(unit="step") as pbar:
        while True:
            if step % cfg["eval_every"] == 0:
                accs = [eval_fn(params, jnp.array(x), jnp.array(y)) for x, y in val_loader]
                val_acc = float(jnp.mean(jnp.array(accs)))
                wandb.log({"val/val_acc": val_acc}, step=step)
                wandb.log({"training_time_s": time.time() - t0}, step=step)
                pbar.set_postfix(val_acc=f"{val_acc:.4f}", best=f"{best_val_acc:.4f}")
                if val_acc > best_val_acc:
                    best_val_acc, best_params, best_step = val_acc, params, step
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= cfg["patience"]:
                    print(f"Early stop at step {step}")
                    break

            x, y   = next(train_iter)
            params, loss = step_fn(params, jnp.array(x), jnp.array(y))
            wandb.log({"train/loss": float(loss)}, step=step)
            step += 1
            pbar.update(1)
            if cfg["max_steps"] and step >= cfg["max_steps"]:
                break

    total_time = time.time() - t0
    test_accs  = [eval_fn(best_params, jnp.array(x), jnp.array(y)) for x, y in test_loader]
    final_acc  = float(jnp.mean(jnp.array(test_accs)))
    print(f"Test Acc: {final_acc:.4f}  Time: {total_time:.1f}s")
    wandb.log({"test/test_acc": final_acc}, step=step)
    wandb.log({"total_training_time_s": total_time}, step=step)
    wandb.finish()
    return final_acc, best_val_acc, best_step, total_time


if __name__ == "__main__":
    cfg = yaml.safe_load(open(CFG_PATH))
    base_key = jax.random.key(cfg["random_seed"])
    all_keys = jax.random.split(base_key,
        len(cfg["batch_sizes"]) * len(cfg["tasks"]) * cfg["num_runs"])
    ki = 0

    for batch_size in cfg["batch_sizes"]:
        for task_cfg in cfg["tasks"]:
            print(f"\n{'='*50}\n{FRAMEWORK} | {task_cfg['name']} | bs={batch_size}")
            for run_number in range(1, cfg["num_runs"] + 1):
                run(cfg, task_cfg, batch_size, run_number, all_keys[ki])
                ki += 1
                gc.collect()
                jax.clear_caches()
