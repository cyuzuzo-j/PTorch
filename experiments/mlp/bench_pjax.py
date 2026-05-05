##################################################
###   Benchmark — pjax_orr                     ###
##################################################
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))

import gc
import yaml
import jax
import jax.numpy as jnp
import pandas as pd
import tqdm, time

import pjax_orr
import pjax_orr.nn as nn
import pjax_orr.optim as optim
import pjax_orr.config as config

reshape = getattr(pjax_orr, 'reshape', jnp.reshape)
cross_entropy = pjax_orr.core.api.cross_entropy

FRAMEWORK = "pjax"
CFG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
from experiments.shared.data import MNISTDataModule, InfiniteCifarDataModule
DATASETS = {"MNIST": MNISTDataModule, "CIFAR10": InfiniteCifarDataModule}


class MLPBase:
    def __init__(self, hidden, in_features, classes):
        self.hidden = hidden
        self.in_features = in_features
        self.classes = classes

    def build(self):
        class MLP(nn.Module):
            def __init__(self, hidden, in_features, classes):
                super().__init__()
                last = in_features
                for i, f in enumerate(hidden):
                    setattr(self, f"linear_{i}", nn.Linear(last, f))
                    setattr(self, f"relu_{i}",   nn.ReLU(f))
                    last = f
                self.n_hidden = len(hidden)
                self.out = nn.Linear(last, classes)

            def get_params(self, key, init_x=None):
                try:
                    if init_x is None:
                        return self.init(key)
                    return self.init(key, init_x)
                except TypeError:
                    try:
                        return self.init(key)
                    except Exception:
                        return self.init(key)

            def __call__(self, x):
                x = reshape(x, (x.shape[0], -1))
                for i in range(self.n_hidden):
                    x = getattr(self, f"linear_{i}")(x)
                    x = getattr(self, f"relu_{i}")(x)
                return self.out(x)

        return MLP(self.hidden, self.in_features, self.classes)


def run(cfg, task_cfg, batch_size, run_number, key):
    OPTIM_MODULES = {**vars(optim)}

    seed = cfg.get("random_seed", 0)

    dataset_name = task_cfg.get("dataset", task_cfg["name"])
    dataset_cls = DATASETS[dataset_name]
    data_seed   = int(jax.random.randint(key, (), 0, 2**30))
    ds = dataset_cls(batch_size=batch_size, seed=data_seed)
    train_iter = ds.train_iterator()
    val_loader  = ds.val_dataloader()
    test_loader = ds.test_dataloader()

    model_key, _ = jax.random.split(key)
    model = MLPBase(task_cfg["hidden"], task_cfg["in_features"], task_cfg["classes"]).build()
    params = model.init(model_key)

    # pick optimizer name from config
    opt_name = cfg.get("pjax_optimizer") or cfg.get("pjax_orr_optimizer")
    opt_kwargs = cfg.get("pjax_optimizer_kwargs") or cfg.get("pjax_orr_optimizer_kwargs") or {}
    optimizer = OPTIM_MODULES[opt_name](**opt_kwargs)

    results_dir = os.path.join(os.path.dirname(__file__), cfg.get("results_dir", "results"))
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"{FRAMEWORK}_{task_cfg['name']}.csv")
    csv_rows = []

    @jax.jit
    def step_fn(params, x, y):
        def apply_fn(p):
            pred = model.apply(p, x)
            y_oh = jax.nn.one_hot(y, pred.shape[-1])
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

    # Compile step
    print("Compiling...")
    dummy_x, dummy_y = next(train_iter)
    _, _loss = step_fn(params, jnp.array(dummy_x), jnp.array(dummy_y))
    _loss.block_until_ready()
    dummy_val_x, dummy_val_y = next(iter(val_loader))
    _acc = eval_fn(params, jnp.array(dummy_val_x), jnp.array(dummy_val_y))
    _acc.block_until_ready()
    print("Compilation done.")

    best_val_acc, best_params, best_step = 0.0, params, 0
    no_improve = 0
    step = 0
    t0 = time.time()

    with tqdm.tqdm(unit="step") as pbar:
        while True:
            if step % cfg["eval_every"] == 0:
                accs = [eval_fn(params, jnp.array(x), jnp.array(y)) for x, y in val_loader]
                val_acc = float(jnp.mean(jnp.array(accs)))
                elapsed = time.time() - t0
                csv_rows.append({
                    "framework": FRAMEWORK, "task": task_cfg["name"],
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

            x, y   = next(train_iter)
            params, loss = step_fn(params, jnp.array(x), jnp.array(y))
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
        "run": run_number, "step": step,
        "val_acc": final_acc, "wall_time_s": total_time,
    })

    # Write / append CSV
    df = pd.DataFrame(csv_rows)
    df.to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)
    print(f"Results appended to {csv_path}")
    return final_acc, best_val_acc, best_step, total_time


def main():
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


if __name__ == '__main__':
    main()
