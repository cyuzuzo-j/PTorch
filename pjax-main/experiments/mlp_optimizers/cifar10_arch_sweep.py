################################################
###   CIFAR10 Architecture Sweep             ###
################################################
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import gc
import torch
import torch.nn as tnn
import numpy as np
from ptorch.nn.modules import LinearBias, ReLU
from ptorch.core.ops import CrossEntropyProjection
from ptorch.optim_static import AlternatingProjections
from experiments.shared.data import InfiniteCifarDataModule
import tqdm
import time

from aim import Run
from faker import Faker
fake = Faker()

# ─── Configuration ────────────────────────────────────────────────────────────
BATCH_SIZE = 512
RANDOM_SEED = 42
MAX_STEPS = 5000
NUM_RUNS = 1

ARCHITECTURES = [
    [4096], 
    [2048, 2048],
    [1024, 1024, 1024],
    [512, 512, 512, 512],
    [256, 256, 256, 256, 256, 256],
    [128, 128, 128, 128, 128, 128, 128, 128],
    [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
]

class MLP_ptorch(tnn.Module):
    def __init__(self, hidden_features, in_features, classes, skip=False):
        super().__init__()
        self.hidden_features = hidden_features
        self.skip = skip

        last_f = in_features
        self.layers = tnn.ModuleList()
        for f in hidden_features:
            self.layers.append(LinearBias(last_f, f))
            self.layers.append(ReLU(f))
            last_f = f

        out_features = sum(hidden_features) if skip else hidden_features[-1]
        self.out = LinearBias(out_features, classes)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        for i in range(0, len(self.layers), 2):
            x = self.layers[i](x)      # LinearBias
            x = self.layers[i + 1](x)  # ReLU
        return self.out(x)


def run_architecture(hidden_features, device, eval_every=100, patience=10, max_steps=None, run_number=1, batch_size=512):
    torch.manual_seed(RANDOM_SEED + run_number)

    # Setup Data
    dataset = InfiniteCifarDataModule(batch_size=batch_size, seed=RANDOM_SEED)
    train_iter = dataset.train_iterator()
    val_loader = dataset.val_dataloader()
    test_loader = dataset.test_dataloader()

    # Setup Model & Optimizer
    model = MLP_ptorch(hidden_features, 3 * 32 * 32, 10).to(device)
    optimizer = AlternatingProjections(model.parameters())

    # Aim Run Init
    name_int = np.random.randint(0, 100)
    num_layers = len(hidden_features)
    arch_str = "-".join(map(str, hidden_features))
    run_name = f"CIFAR10 {num_layers}L_{arch_str} {name_int}"
    run = Run(experiment="CIFAR10 Arch Sweep")
    run.name = run_name
    
    hparams = {
        "task": "CIFAR10",
        "model": "MLP_ptorch",
        "optimizer": "AlternatingProjections",
        "framework": "ptorch",
        "batch_size": batch_size,
        "seed": RANDOM_SEED,
        "eval_every": eval_every,
        "patience": patience,
        "max_steps": max_steps,
        "run_number": run_number,
        "hidden_features": hidden_features,
        "num_layers": num_layers,
        "architecture": arch_str,
    }
    run["hparams"] = hparams

    def step_fn(model, optimizer, x, y):
        logits = model(x)
        y_one_hot = torch.nn.functional.one_hot(y.long(), num_classes=logits.shape[-1]).float()

        projected = CrossEntropyProjection.apply(logits, y_one_hot)

        optimizer.zero_grad()
        projected.sum().backward()

        loss = torch.nn.functional.cross_entropy(logits.detach(), y.long())
        optimizer.step()
        return loss

    def eval_fn(model, x, y):
        with torch.no_grad():
            pred = model(x)
            return (pred.argmax(dim=-1) == y).float().mean()

    best_val_acc, best_step = 0.0, 0
    no_improve_cycles = 0
    history = []

    print(f"Starting training... Layers: {num_layers}, Width: {hidden_features[0]}")
    step = 0
    train_start_time = time.time()
    with tqdm.tqdm(unit="step") as pbar:
        while True:
            if step % eval_every == 0:
                model.eval()
                accs = [eval_fn(model, torch.tensor(x, dtype=torch.float32, device=device),
                                torch.tensor(y, dtype=torch.long, device=device))
                        for x, y in val_loader]
                val_acc = float(torch.stack(accs).mean())
                history.append((step, val_acc))
                model.train()

                elapsed_time = time.time() - train_start_time
                run.track(val_acc, name="val_acc", step=step, context={"subset": "val"})
                run.track(best_val_acc, name="best_val_acc", step=step, context={"subset": "val"})
                run.track(elapsed_time, name="training_time_s", step=step)

                pbar.set_postfix(val_acc=f"{val_acc:.4f}", best=f"{best_val_acc:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc, best_step = val_acc, step
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    no_improve_cycles = 0
                else:
                    no_improve_cycles += 1

                if no_improve_cycles >= patience:
                    print(f"Early stopping at step {step}")
                    break

            x, y = next(train_iter)
            x = torch.tensor(x, dtype=torch.float32, device=device)
            y = torch.tensor(y, dtype=torch.long, device=device)
            loss = step_fn(model, optimizer, x, y)

            run.track(float(loss), name="loss", step=step, context={"subset": "train"})

            step += 1
            pbar.update(1)
            if max_steps and step >= max_steps:
                break

    total_train_time = time.time() - train_start_time
    if 'best_state' in locals():
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

    return final_acc


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for hidden_features in ARCHITECTURES:
        print(f"\n\n{'='*50}")
        print(f"Architecture: {hidden_features}")
        print(f"{'='*50}")

        for run_number in range(1, NUM_RUNS + 1):
            run_architecture(
                hidden_features, device,
                eval_every=100, max_steps=MAX_STEPS,
                run_number=run_number, batch_size=BATCH_SIZE,
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
