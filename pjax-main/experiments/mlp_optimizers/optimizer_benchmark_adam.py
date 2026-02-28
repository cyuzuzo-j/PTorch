################################################
###   Optimizer benchmark for Adam (PyTorch) ###
###   Standard backprop baseline             ###
################################################
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import gc
import torch
import torch.nn as tnn
import torch.nn.functional as F
import numpy as np
from experiments.shared.data import (
    MNISTDataModule,
    InfiniteCifarDataModule
)
import tqdm
import time

from aim import Run

## generate experiment names
from faker import Faker
fake = Faker()


# ─── Configuration ────────────────────────────────────────────────────────────
BATCH_SIZES = [32, 128, 512, 2048]
RANDOM_SEED = 42
MAX_STEPS = 500
NUM_RUNS = 2


# ─── Standard PyTorch MLP ─────────────────────────────────────────────────────
class MLP_torch(tnn.Module):
    """Standard MLP using regular PyTorch layers.

    Same architecture as MLP_ptorch for fair comparison.
    """

    def __init__(self, hidden_features, in_features, classes, skip=True):
        super().__init__()
        self.hidden_features = hidden_features
        self.skip = skip

        last_f = in_features
        self.layers = tnn.ModuleList()
        for f in hidden_features:
            self.layers.append(tnn.Linear(last_f, f))
            self.layers.append(tnn.ReLU())
            last_f = f

        out_features = sum(hidden_features) if skip else hidden_features[-1]
        self.out = tnn.Linear(out_features, classes)

    def forward(self, x):
        # Flatten input for MLP
        x = x.reshape(x.shape[0], -1)
        xs = []
        for i in range(0, len(self.layers), 2):
            x = self.layers[i](x)      # Linear
            x = self.layers[i + 1](x)  # ReLU
            xs.append(x)

        if self.skip:
            x = torch.cat(xs, dim=-1)

        return self.out(x)


# ─── Task definitions ────────────────────────────────────────────────────────
tasks = [
    {
        "name": "MNIST",
        "dataset": MNISTDataModule,
        "model_fn": lambda: MLP_torch([512], 28 * 28, 10),
    },
    {
        "name": "CIFAR10",
        "dataset": InfiniteCifarDataModule,
        "model_fn": lambda: MLP_torch([1024], 3 * 32 * 32, 10),
    },
]


def run_task(task, device, eval_every=100, patience=10, max_steps=None,
             run_number=1, batch_size=2048):
    """Run a single training experiment with Aim tracking."""

    torch.manual_seed(RANDOM_SEED + run_number)

    # Setup Data
    dataset = task["dataset"](batch_size=batch_size, seed=RANDOM_SEED)
    train_iter = dataset.train_iterator()
    val_loader = dataset.val_dataloader()
    test_loader = dataset.test_dataloader()

    # Setup Model & Optimizer
    model = task["model_fn"]().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    # Aim Run Init — batch_size visible in experiment name for quick comparison
    name_int = np.random.randint(0, 100)
    run = Run(experiment=f"{fake.name()} {name_int}")
    hparams = {
        "task": task["name"],
        "model": model.__class__.__name__,
        "optimizer": "Adam",
        "framework": "torch",              # ← key for comparing pjax vs ptorch vs torch
        "batch_size": batch_size,
        "seed": RANDOM_SEED,
        "eval_every": eval_every,
        "patience": patience,
        "max_steps": max_steps,
        "run_number": run_number,
    }
    run["hparams"] = hparams

    def step_fn(model, optimizer, x, y):
        """One training step using standard cross-entropy + Adam."""
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y.long())
        loss.backward()
        optimizer.step()
        return loss

    def eval_fn(model, x, y):
        with torch.no_grad():
            pred = model(x)
            return (pred.argmax(dim=-1) == y).float().mean()

    # Training Loop
    best_val_acc, best_step = 0.0, 0
    no_improve_cycles = 0
    history = []

    print(f"Starting training... Eval every {eval_every}, Patience {patience}")
    step = 0
    train_start_time = time.time()
    with tqdm.tqdm(unit="step") as pbar:
        while True:
            # Eval
            if step % eval_every == 0:
                model.eval()
                accs = [eval_fn(model, torch.tensor(x, dtype=torch.float32, device=device),
                                torch.tensor(y, dtype=torch.long, device=device))
                        for x, y in val_loader]
                val_acc = float(torch.stack(accs).mean())
                history.append((step, val_acc))
                model.train()

                # Aim tracking
                elapsed_time = time.time() - train_start_time
                run.track(val_acc, name="val_acc", step=step,
                          context={"subset": "val"})
                run.track(best_val_acc, name="best_val_acc", step=step,
                          context={"subset": "val"})
                run.track(elapsed_time, name="training_time_s", step=step)

                pbar.set_postfix(val_acc=f"{val_acc:.4f}",
                                 best=f"{best_val_acc:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc, best_step = val_acc, step
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    no_improve_cycles = 0
                else:
                    no_improve_cycles += 1

                if no_improve_cycles >= patience:
                    print(f"Early stopping at step {step}")
                    break

            # Train
            x, y = next(train_iter)
            x = torch.tensor(x, dtype=torch.float32, device=device)
            y = torch.tensor(y, dtype=torch.long, device=device)
            loss = step_fn(model, optimizer, x, y)

            # Track training loss
            run.track(float(loss), name="loss", step=step,
                      context={"subset": "train"})

            step += 1
            pbar.update(1)
            if max_steps and step >= max_steps:
                break

    # Final Test — use best model
    total_train_time = time.time() - train_start_time
    if 'best_state' in dir():
        model.load_state_dict(best_state)
    model.eval()
    test_accs = [eval_fn(model, torch.tensor(x, dtype=torch.float32, device=device),
                         torch.tensor(y, dtype=torch.long, device=device))
                 for x, y in test_loader]
    final_acc = float(torch.stack(test_accs).mean())
    print(f"Final Test Acc: {final_acc:.4f} | Training Time: {total_train_time:.2f}s")

    run.track(final_acc, name="test_acc", step=step, context={"subset": "test"})
    run.track(total_train_time, name="total_training_time_s", step=step)
    run.close()

    return (final_acc, best_val_acc, best_step, total_train_time, history, step)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for batch_size in BATCH_SIZES:
        for task_info in tasks:
            print(f"\n\n{'='*50}")
            print(f"Running Task: {task_info['name']} | Batch Size: {batch_size}")
            print(f"{'='*50}")

            for run_number in range(1, NUM_RUNS + 1):
                print(f"\n--- Batch: {batch_size} | Run {run_number}/{NUM_RUNS} ---")
                results = run_task(
                    task_info, device,
                    eval_every=50, max_steps=MAX_STEPS,
                    run_number=run_number, batch_size=batch_size,
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
