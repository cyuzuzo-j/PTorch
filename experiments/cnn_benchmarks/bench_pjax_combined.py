##################################################
###   Benchmark — pjax (combined)              ###
###   Unified runner for `pjax` and `pjax_orr`  ###
##################################################
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))

import gc
import yaml
import jax
import jax.numpy as jnp
import importlib
import argparse
import pandas as pd
import tqdm, time

from experiments.cnn_benchmarks.models import SimpleCNN_PJAX

CFG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

DATASETS = {"MNIST": None, "CIFAR10": None}
try:
    from experiments.shared.data import MNISTDataModule, InfiniteCifarDataModule
    DATASETS = {"MNIST": MNISTDataModule, "CIFAR10": InfiniteCifarDataModule}
except Exception:
    pass

def load_impl(name):
    mod = importlib.import_module(name)
    nn = importlib.import_module(f"{name}.nn")
    optim = importlib.import_module(f"{name}.optim")
    try:
        optim_static = importlib.import_module(f"{name}.optim_static")
    except ImportError:
        optim_static = None
    
    try:
        cross_entropy = importlib.import_module(f"{name}.nn").cross_entropy
    except AttributeError:
        cross_entropy = getattr(mod, 'cross_entropy', None)

    return {
        'mod': mod,
        'nn': nn,
        'optim': optim,
        'optim_static': optim_static,
        'cross_entropy': cross_entropy,
    }


def run(cfg, task_cfg, batch_size, run_number, key, impl_name,
        norm='l2', opt_name=None, opt_kwargs=None, loss_name='CrossEntropy', use_muon_activations=False):
    impl = load_impl(impl_name)
    optim = impl['optim']
    optim_static = impl['optim_static']
    cross_entropy = impl['cross_entropy']

    OPTIM_MODULES = {}
    if optim_static is not None:
        OPTIM_MODULES.update({**vars(optim_static)})
    OPTIM_MODULES.update({**vars(optim)})

    FRAMEWORK = impl_name

    seed = cfg.get("random_seed", 0)

    dataset_cls = DATASETS[task_cfg["name"]]
    data_seed   = int(jax.random.randint(key, (), 0, 2**30))

    if task_cfg["name"] == "CIFAR10":
        ds = dataset_cls(batch_size=batch_size, data_dir="./dataset", seed=data_seed)
    else:
        ds = dataset_cls(batch_size=batch_size, seed=data_seed)
        
    train_iter = ds.train_iterator()
    val_loader  = ds.val_dataloader()
    test_loader = ds.test_dataloader()

    model_key, _ = jax.random.split(key)
    
    # Init CNN properly
    model = SimpleCNN_PJAX(
        classes=task_cfg["classes"], 
        in_channels=task_cfg.get("in_channels", 3)
    )
    
    # Dummy forward to initialize params in pjax
    dummy_res = 32 if task_cfg.get("in_channels", 3) == 3 else 28
    dummy_x = jnp.zeros((1, dummy_res, dummy_res, task_cfg.get("in_channels", 3)))
    params = model.init(model_key, dummy_x)

    # Pick optimizer name from config, try both possible key names if not passed
    if opt_name is None:
        opt_name = cfg.get(f"{impl_name}_optimizer") or cfg.get("pjax_optimizer") or cfg.get("pjax_orr_optimizer")
    if opt_kwargs is None:
        opt_kwargs = cfg.get(f"{impl_name}_optimizer_kwargs") or cfg.get("pjax_optimizer_kwargs") or cfg.get("pjax_orr_optimizer_kwargs") or {}
    
    optimizer = OPTIM_MODULES[opt_name](**opt_kwargs)

    # Build a tag that distinguishes this run in the CSV
    tag = f"{FRAMEWORK}_norm={norm}_opt={opt_name}_loss={loss_name}_muon={use_muon_activations}"

    results_dir = os.path.join(os.path.dirname(__file__), cfg.get("results_dir", "results"))
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"{tag}_{task_cfg['name']}.csv")
    csv_rows = []

    # Fall back cross_entropy implementation if missing
    if cross_entropy is None:
        def ce_loss(pred, y_oh):
            return -jnp.sum(y_oh * jax.nn.log_softmax(pred), axis=-1)
        cross_entropy = ce_loss

    @jax.jit
    def step_fn(params, x, y):
        def apply_fn(p):
            pred = model.apply(p, x)
            y_oh = jax.nn.one_hot(y, num_classes=pred.shape[-1])
            return cross_entropy(pred, y_oh)
        loss = jnp.mean(apply_fn(params))
        res = optimizer.update(apply_fn, params)
        if isinstance(res, (tuple, list)):
            new_params = res[0]
        else:
            new_params = res
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

    with tqdm.tqdm(total=cfg["max_steps"], unit="step", desc=f"{task_cfg['name']} | norm={norm} | {opt_name} | {loss_name}") as pbar:
        while step < cfg["max_steps"]:
            # ── Fetch a new batch ─────────────────────────────────────────────
            x_np, y_np = next(train_iter)
            params, loss = step_fn(params, jnp.array(x_np), jnp.array(y_np))

            if step % cfg["eval_every"] == 0:
                accs = [eval_fn(params, jnp.array(x), jnp.array(y)) for x, y in val_loader]
                val_acc = float(jnp.mean(jnp.array(accs)))
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
                    best_val_acc, best_params, best_step = val_acc, params, step
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= cfg["patience"]:
                    print(f"Early stop at step {step}")
                    break

            step += 1
            pbar.update(1)
            if cfg["max_steps"] and step >= cfg["max_steps"]:
                break

    total_time = time.time() - t0
    test_accs  = [eval_fn(best_params, jnp.array(x), jnp.array(y)) for x, y in test_loader]
    final_acc  = float(jnp.mean(jnp.array(test_accs)))
    print(f"Test Acc: {final_acc:.4f}  Time: {total_time:.1f}s")
    
    csv_rows.append({
        "framework": FRAMEWORK, "task": task_cfg["name"],
        "norm": norm, "optimizer": opt_name, "loss": loss_name,
        "use_muon_activations": use_muon_activations,
        "run": run_number, "step": step,
        "val_acc": final_acc, "wall_time_s": total_time,
    })

    # Write / append CSV
    df = pd.DataFrame(csv_rows)
    df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)
    print(f"Results appended to {csv_path}")

    return final_acc, best_val_acc, best_step, total_time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--impl', choices=['pjax', 'pjax_orr'], default=os.environ.get('PJAX_IMPL', 'pjax_orr'))
    p.add_argument('--config', default=CFG_PATH, help="Path to YAML config file")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    base_key = jax.random.key(cfg["random_seed"])
    
    # Calculate total runs
    num_optimizers = len(cfg.get("optimizers", [{}]))
    all_keys = jax.random.split(base_key,
        len(cfg["batch_sizes"]) * len(cfg["tasks"]) * cfg["num_runs"] * num_optimizers)
    ki = 0

    # Standardize values for plotting comparison
    norm = "l2"
    loss_name = "CrossEntropy"
    muon_act = False

    # Optimizer sweep fallback
    impl_key = f"{args.impl}_optimizer"
    kwargs_key = f"{args.impl}_optimizer_kwargs"
    optimizers = cfg.get("optimizers", [
        {"name": cfg.get(impl_key, "DouglasRachford"),
         "kwargs": cfg.get(kwargs_key, {})}
    ])

    for batch_size in cfg["batch_sizes"]:
        for task_cfg in cfg["tasks"]:
            for opt_entry in optimizers:
                opt_name   = opt_entry["name"]
                opt_kwargs = opt_entry.get("kwargs", {})
                header = (f"{args.impl} | {task_cfg['name']} "
                          f"| norm={norm} | opt={opt_name} | loss={loss_name} | muon={muon_act} | bs={batch_size}")
                print(f"\n{'='*60}\n{header}")
                for run_number in range(1, cfg["num_runs"] + 1):
                    run(cfg, task_cfg, batch_size, run_number, all_keys[ki], args.impl,
                        norm=norm, opt_name=opt_name, opt_kwargs=opt_kwargs, loss_name=loss_name, use_muon_activations=muon_act)
                    ki += 1
                    gc.collect()
                    jax.clear_caches()


if __name__ == '__main__':
    main()
