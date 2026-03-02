################################################
###   Test suite for all of the optimizers   ###
################################################
import gc
import jax
import jax.numpy as jnp
import numpy as np
import pjax
from pjax import nn, optim
from data import (
    MNISTDataModule,
    CIFAR10DataModule
) 
import tqdm
import time
from mlp import MLP_pjax
from cnn import CNN_pjax
from aim import Run

## generate experiment names (so that its easy to refer to them)
from faker import Faker
fake = Faker()


BATCH_SIZE = 128
RANDOM_SEED = 42
PROJECTION_STEPS = 50
MAX_STEPS = 10_000
NUM_RUNS = 3
jax_random_key = jax.random.key(RANDOM_SEED)

tasks = [
    {
        "name":"MNIST",
        "dataset":MNISTDataModule,
        "model":MLP_pjax([256],28*28,10),
        "vectorise":True
    },
]

optimizers = [
    {
        "name": "Dykstra",
        "optimizer": optim.Dykstra(steps_per_update=PROJECTION_STEPS),
    }
]

def run_task(task, opt_info, jax_random_key, eval_every=100, patience=10, max_steps=None, run_number=1):
    # Split key into independent sub-keys for data, model init, and naming
    data_key, model_key, name_key = jax.random.split(jax_random_key, 3)
    
    # Setup Data
    data_seed = int(jax.random.randint(data_key, (), 0, 2**30))
    dataset = task["dataset"](batch_size=BATCH_SIZE, seed=data_seed)
    train_iter = dataset.train_iterator()
    val_loader = dataset.val_dataloader()
    test_loader = dataset.test_dataloader()
    
    # Setup Model & Params
    init_x = jnp.array(next(iter(val_loader))[0])
    model = task["model"]
    params = model.get_params(model_key, init_x)

    optimizer = opt_info["optimizer"] 
    
    # Aim Run Init
    name_int = int(jax.random.randint(name_key, (), 0, 100))
    run = Run(experiment=f"{fake.name()} {name_int}")
    run["hparams"] = {
        "task": task["name"],
        "model": model.__class__.__name__,
        "optimizer": opt_info['name'],
        "batch_size": BATCH_SIZE,
        "seed": RANDOM_SEED,
        "eval_every": eval_every,
        "patience": patience,
        "max_steps": max_steps,
        "projection_steps": PROJECTION_STEPS,
        "run_number": run_number,
    }

    opt_state = None

    def step_fn(params, opt_state, x, y):
        def apply_fn(cur_params):
            pred = model.apply(cur_params, x)
            y_one_hot = jax.nn.one_hot(y, pred.shape[-1])
            return pjax.cross_entropy(pred, y_one_hot)

        loss = apply_fn(params)
        loss = jnp.mean(loss)
        new_params = optimizer.update(apply_fn, params)[0]
        return new_params, opt_state, loss

    def eval_fn(params, x, y):
        pred = model.apply(params, x)
        if not isinstance(pred, jnp.ndarray) and hasattr(pred, 'value'):
            pred = pred.value
        return jnp.mean(jnp.argmax(pred, axis=-1) == y)

    # Training Loop
    steps_per_update = getattr(optimizer, "steps_per_update", 1)
    best_val_acc, best_params, best_step = 0.0, params, 0
    no_improve_cycles = 0
    history = []
    
    print(f"Starting training... Eval every {eval_every}, Patience {patience}")
    step = 0
    train_start_time = time.time()
    with tqdm.tqdm(unit="step") as pbar:
        while True:
            # Eval
            if step % eval_every == 0:
                accs = [eval_fn(params, jnp.array(x), jnp.array(y)) for x, y in val_loader]
                val_acc = float(jnp.mean(jnp.array(accs)))
                history.append((step, val_acc))
                
                # Aim Tracking logic
                elapsed_time = time.time() - train_start_time
                run.track(val_acc, name="val_acc", step=step, context={"subset": "val"})
                run.track(best_val_acc, name="best_val_acc", step=step, context={"subset": "val"})
                run.track(elapsed_time, name="training_time_s", step=step)
                
                pbar.set_postfix(val_acc=f"{val_acc:.4f}", best=f"{best_val_acc:.4f}")
                
                if val_acc > best_val_acc:
                    best_val_acc, best_params, best_step = val_acc, params, step
                    no_improve_cycles = 0
                else:
                    no_improve_cycles += 1
                    if "decay" in opt_info and opt_info["decay"] == True:
                        optimizer.relaxation *= 0.9
                        print(f"  Decay: relaxation -> {optimizer.relaxation:.4f}")
                        run.track(optimizer.relaxation, name="relaxation", step=step)
                
                if no_improve_cycles >= patience:
                    print(f"Early stopping at step {step}")
                    break

            # Train
            x, y = next(train_iter)
            params, opt_state, loss = step_fn(params, opt_state, jnp.array(x), jnp.array(y))
            
            # Track training loss
            run.track(float(loss), name="loss", step=step, context={"subset": "train"})

            step += steps_per_update
            pbar.update(steps_per_update)
            if max_steps and step >= max_steps:
                break

    # Final Test
    total_train_time = time.time() - train_start_time
    test_accs = [eval_fn(best_params, jnp.array(x), jnp.array(y)) for x, y in test_loader]
    final_acc = float(jnp.mean(jnp.array(test_accs)))
    print(f"Final Test Acc: {final_acc:.4f} | Training Time: {total_train_time:.2f}s")
    
    run.track(final_acc, name="test_acc", step=step, context={"subset": "test"})
    run.track(total_train_time, name="total_training_time_s", step=step)
    run.close()
    
    return (final_acc, best_val_acc, best_step, total_train_time, history, step)


if __name__ == "__main__":
    # Pre-split independent keys for each (task, optimizer, run) combination
    num_total_runs = len(tasks) * len(optimizers) * NUM_RUNS
    all_keys = jax.random.split(jax_random_key, num_total_runs)
    key_idx = 0
    
    for task_info in tasks:
        print(f"\n\n{'='*50}")
        print(f"Running Task: {task_info['name']}")
        print(f"{'='*50}")
        
        for opt_info in optimizers:
            for run_number in range(1, NUM_RUNS + 1):
                print(f"\n--- Optimizer: {opt_info['name']} | Run {run_number}/{NUM_RUNS} ---")
                run_key = all_keys[key_idx]
                key_idx += 1
                results = run_task(task_info, opt_info, run_key, eval_every=50, max_steps=MAX_STEPS, run_number=run_number)
                gc.collect()
                jax.clear_caches()
                from pjax.core.computation import vmap_ids_order
                vmap_ids_order.clear()
                from pjax.optim import prune_shape_transforms
                prune_shape_transforms.cache_clear()
