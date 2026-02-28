################################################
###   Optimizer benchmark for ptorch         ###
###   (PyTorch alternating projections)      ###
################################################
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import gc
import torch
import torch.nn as tnn
import numpy as np
from ptorch.nn.modules import LinearBias, ReLU, Step
from ptorch.core.ops import CrossEntropyProjection, MarginLossProjection
from ptorch.optim_static import (
    ProjectionMuon,
    AlternatingProjections,
    AlternatingProjectionsMomentum,
    ProjectionSGD,
    ProjectionAdam,
    ProjectionAdagrad,
    ProjectionAdadelta,
)
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


# ─── Configuration ───────────────────────────────────────────────────────────────────
BATCH_SIZES = [32]
RANDOM_SEED = 42
MAX_STEPS = 5000
NUM_RUNS = 2


# ─── Alpha schedule helpers ──────────────────────────────────────────────────────
def make_alphas(n_layers: int, alpha_top: float, decay_rate: float) -> list:
    """
    Exponentially decayed alpha schedule.
    Index 0 = input-side layer  → alpha ≈ 0
    Index n-1 = output layer    → alpha_top

    alpha[i] = alpha_top * exp(-decay_rate * (n_layers - 1 - i))
    Setting decay_rate=0 gives a flat (no-decay) schedule.
    """
    return [alpha_top * float(np.exp(-decay_rate * (n_layers - 1 - i)))
            for i in range(n_layers)]


# ─── PyTorch MLP using ptorch projection layers ──────────────────────────────
class MLP_ptorch(tnn.Module):
    """MLP using ptorch projection layers (LinearBias + ReLU).

    Mirrors the pjax MLP_pjax architecture for fair comparison.
    """

    def __init__(self, hidden_features, in_features, classes, skip=False,
                 alpha_top: float = 1.0, decay_rate: float = 0.0, num_iters: int = 1):
        super().__init__()
        self.hidden_features = hidden_features
        self.skip = skip
        self.num_iters = num_iters

        n_linear = len(hidden_features) + 1
        alphas = make_alphas(n_linear, alpha_top=alpha_top, decay_rate=decay_rate)
        self.alphas = alphas

        last_f = in_features
        self.layers = tnn.ModuleList()
        for layer_idx, f in enumerate(hidden_features):
            self.layers.append(LinearBias(last_f, f, alpha=alphas[layer_idx], num_iters=num_iters))
            self.layers.append(ReLU(f))
            last_f = f

        out_features = sum(hidden_features) if skip else hidden_features[-1]
        self.out = LinearBias(out_features, classes, alpha=alphas[-1], num_iters=num_iters)

    def forward(self, x):
        # Flatten input for MLP
        x = x.reshape(x.shape[0], -1)
        xs = []
        for i in range(0, len(self.layers), 2):
            x = self.layers[i](x)      # LinearBias
            x = self.layers[i + 1](x)  # ReLU
            xs.append(x)

        return self.out(x)


# ─── Task definitions ────────────────────────────────────────────────────────
tasks = [
    {
        "name": "MNIST",
        "dataset": MNISTDataModule,
        "model_fn": lambda alpha_top=1.0, decay_rate=0.0, num_iters=1: MLP_ptorch(
            [128,128], 28 * 28, 10,
            alpha_top=alpha_top, decay_rate=decay_rate, num_iters=num_iters
        ),
    },
    
]


def run_task(task, device, optimizer_class, optimizer_kwargs={}, eval_every=100, patience=10, max_steps=None,
             run_number=1, batch_size=2048, alpha_top=1.0, decay_rate=0.0, num_iters=1):
    """Run a single training experiment with Aim tracking."""

    torch.manual_seed(RANDOM_SEED + run_number)

    # Setup Data
    dataset = task["dataset"](batch_size=batch_size, seed=RANDOM_SEED)
    train_iter = dataset.train_iterator()
    val_loader = dataset.val_dataloader()
    test_loader = dataset.test_dataloader()

    # Setup Model & Optimizer
    model = task["model_fn"](alpha_top=alpha_top, decay_rate=decay_rate, num_iters=num_iters).to(device)
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    # Aim Run Init — batch_size visible in experiment name for quick comparison
    name_int = np.random.randint(0, 100)
    run = Run(experiment=f"{fake.name()} {name_int}")
    hparams = {
        "task": task["name"],
        "model": model.__class__.__name__,
        "hidden_layers": getattr(model, "hidden_features", None),
        "optimizer": optimizer_class.__name__,
        "learning_rate": optimizer_kwargs.get("lr", 1.0),
        "framework": "ptorch",            # ← key for comparing pjax vs ptorch
        "batch_size": batch_size,
        "seed": RANDOM_SEED,
        "eval_every": eval_every,
        "patience": patience,
        "max_steps": max_steps,
        "run_number": run_number,
        "activation": "Step",
        "alpha_top": alpha_top,
        "decay_rate": decay_rate,
        "num_iters": num_iters,
        "alphas": getattr(model, "alphas", None),
    }
    run["hparams"] = hparams

    def step_fn(model, optimizer, x, y):
        """One training step using cross-entropy projection."""
        logits = model(x)
        y_one_hot = torch.nn.functional.one_hot(y.long(), num_classes=logits.shape[-1]).float()

        # CrossEntropyProjection: forward returns logits, backward projects
        projected = MarginLossProjection.apply(logits, y_one_hot)

        optimizer.zero_grad()
        projected.sum().backward()

        loss = torch.nn.functional.cross_entropy(logits.detach(), y.long())
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

    optimizers_to_test = [
        (ProjectionAdadelta, {'lr': 10_000.0}),
    ]

    # (alpha_top, decay_rate): 1.0/0.0 = flat/no-decay baseline;
    # 1e4/2.0 = strong decay from output to input layer
    alpha_schedules = [
        (1e0, 0.0),   # flat schedule (no decay)
    ]

    num_iters_list = [1, 3, 10]  # 1 = single sequential pass, 3/10 = iterative

    for batch_size in BATCH_SIZES:
        for task_info in tasks:
            for opt_class, opt_kwargs in optimizers_to_test:
                for alpha_top, decay_rate in alpha_schedules:
                    for num_iters in num_iters_list:
                        print(f"\n\n{'='*50}")
                        print(f"Task: {task_info['name']} | Batch: {batch_size} | "
                              f"Opt: {opt_class.__name__} | α_top={alpha_top:.0e} | "
                              f"decay={decay_rate} | num_iters={num_iters}")
                        print(f"{'='*50}")

                        for run_number in range(1, NUM_RUNS + 1):
                            print(f"\n--- Run {run_number}/{NUM_RUNS} ---")
                            results = run_task(
                                task_info, device,
                                optimizer_class=opt_class, optimizer_kwargs=opt_kwargs,
                                eval_every=50, max_steps=MAX_STEPS,
                                run_number=run_number, batch_size=batch_size,
                                alpha_top=alpha_top, decay_rate=decay_rate,
                                num_iters=num_iters,
                            )
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
