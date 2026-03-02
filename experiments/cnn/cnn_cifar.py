import os
import sys
import time
import gc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))

import jax
import jax.numpy as jnp
import torch
import torch.nn as tnn
import torch.optim as toptim
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import pjax
from pjax import nn, optim, optim_eff
from experiments.shared.data import InfiniteCifarLoader
import wandb
from tqdm import tqdm
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

def log(msg):
    print(f"[INFO] {msg}")

# --- Constants ---
BATCH_SIZE = 8
EPOCHS = 3

# Proxy-net widths (matches airbench proxy architecture)
PROXY_WIDTHS = {'block1': 4, 'block2': 8, 'block3': 8}
WHITEN_KERNEL_SIZE = 2
WHITEN_WIDTH = 2 * 3 * WHITEN_KERNEL_SIZE**2  # 24
SCALING_FACTOR = 1 / 9

hyp = {
    'aug': {'flip': True, 'translate': 4, 'cutout': 12},
}

# --- Data Loading ---
log("Loading CIFAR-10 dataset...")
base_train_loader = InfiniteCifarLoader('./dataset', train=True, batch_size=BATCH_SIZE, aug=hyp['aug'])
base_test_loader  = InfiniteCifarLoader('./dataset', train=False, batch_size=BATCH_SIZE)

class EpochWrapper:
    """Wraps an InfiniteCifarLoader to yield one epoch of (images, labels) as numpy arrays."""
    def __init__(self, loader, num_samples):
        self.loader_iter = iter(loader)
        self.steps = num_samples // loader.batch_size

    def __iter__(self):
        for _ in range(self.steps):
            _, images, labels = next(self.loader_iter)
            # loader yields (N, C, H, W) float16 on CUDA → convert to (N, H, W, C) float32 numpy
            yield images.float().permute(0, 2, 3, 1).cpu().numpy(), labels.cpu().numpy()

gc.collect()

# ─── PJAX Model (proxy-net architecture, ReLU, no GELU) ──────────────────────

class ConvGroup(nn.Module):
    """Conv → MaxPool(2×2) → BN → ReLU → Conv → BN → ReLU  (depth-2 proxy-net block)."""
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = nn.Conv2D(channels_in,  channels_out, kernel_shape=(3, 3), padding="SAME")
        self.pool  = nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        #self.bn1   = nn.BatchNorm()
        self.relu1 = nn.ReLU_NB()
        self.conv2 = nn.Conv2D(channels_out, channels_out, kernel_shape=(3, 3), padding="SAME")
        #self.bn2   = nn.BatchNorm()
        self.relu2 = nn.ReLU_NB()

    def __call__(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        #x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        #x = self.bn2(x)
        x = self.relu2(x)
        return x

class CNN_PJAX(nn.Module):
    """PJAX proxy-net: Whiten → ReLU → ConvGroup×3 → MaxPool(3×3) → Linear × (1/9)."""
    def __init__(self, classes=10):
        super().__init__()
        w = PROXY_WIDTHS
        self.whiten = nn.Conv2D(3, WHITEN_WIDTH, kernel_shape=(WHITEN_KERNEL_SIZE, WHITEN_KERNEL_SIZE), padding="VALID")
        self.whiten_act = nn.ReLU_NB()
        self.group1 = ConvGroup(WHITEN_WIDTH,  w['block1'])
        self.group2 = ConvGroup(w['block1'],   w['block2'])
        self.group3 = ConvGroup(w['block2'],   w['block3'])
        self.pool   = nn.MaxPool2D(pool_size=(3, 3), strides=(3, 3))
        self.head   = nn.Linear(2 * 2 * w['block3'], classes)

    def __call__(self, x):
        x = self.whiten(x)
        x = self.whiten_act(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.pool(x)
        x = pjax.reshape(x, (x.shape[0], -1))
        return self.head(x) 

# ─── PyTorch Baseline (same proxy-net architecture, ReLU, no GELU) ───────────

class TConvGroup(tnn.Module):
    """PyTorch equivalent of the PJAX ConvGroup."""
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = tnn.Conv2d(channels_in,  channels_out, kernel_size=3, padding=1, bias=False)
        self.pool  = tnn.MaxPool2d(2)
        self.bn1   = tnn.BatchNorm2d(channels_out)
        self.conv2 = tnn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1, bias=False)
        self.bn2   = tnn.BatchNorm2d(channels_out)

    def forward(self, x):
        x = tnn.functional.relu(self.bn1(self.pool(self.conv1(x))))
        x = tnn.functional.relu(self.bn2(self.conv2(x)))
        return x

class CNN_PyTorch(tnn.Module):
    """PyTorch proxy-net: Whiten → ReLU → TConvGroup×3 → MaxPool(3) → Linear × (1/9)."""
    def __init__(self, classes=10):
        super().__init__()
        w = PROXY_WIDTHS
        self.whiten = tnn.Conv2d(3, WHITEN_WIDTH, kernel_size=WHITEN_KERNEL_SIZE, padding=0, bias=True)
        self.group1 = TConvGroup(WHITEN_WIDTH, w['block1'])
        self.group2 = TConvGroup(w['block1'],  w['block2'])
        self.group3 = TConvGroup(w['block2'],  w['block3'])
        self.pool   = tnn.MaxPool2d(3)
        self.head   = tnn.Linear(2 * 2 * w['block3'], classes, bias=False)

    def forward(self, x):
        x = tnn.functional.relu(self.whiten(x))
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.head(x) * SCALING_FACTOR

# ─── Training Functions ───────────────────────────────────────────────────────

def train_pjax_model(name, optimizer_cls, epochs=EPOCHS):
    log(f"--- Training (PJAX): {name} ---")
    model = CNN_PJAX(classes=10)
    params = model.init(jax.random.key(0))
    optimizer = optimizer_cls(steps_per_update=100)

    aim_run = wandb.init(project="pjax", name=name)
    wandb.config.update({
        "dataset": "CIFAR-10", "batch_size": BATCH_SIZE, "epochs": epochs,
        "model": "CNN_PJAX", "framework": "pjax",
        "optimizer": optimizer_cls.__name__, "steps_per_update": 10,
        "widths": PROXY_WIDTHS, "whiten_width": WHITEN_WIDTH, "scaling_factor": SCALING_FACTOR,
        "jax_backend": jax.default_backend(),
    })

    @jax.jit
    def train_step(params, x, y):
        def apply_fn(params):
            logits = model.apply(params, x)
            y_one_hot = jax.nn.one_hot(y, num_classes=10)  # keep float32 — complex64 broke cross-entropy projection
            return pjax.cross_entropy(logits, y_one_hot)
        return optimizer.update(apply_fn, params)

    @jax.jit
    def eval_step(params, x, y):
        logits = model.apply(params, x)
        preds = jnp.argmax(logits, axis=-1)
        return jnp.mean(preds == y)

    def evaluate(params, loader):
        accs = [float(eval_step(params, jnp.array(x), jnp.array(y))) for x, y in loader]
        return sum(accs) / len(accs) if accs else 0.0

    loss_history, train_acc_history, val_acc_history = [], [], []
    start_time = time.time()
    global_step = 0

    for epoch in range(epochs):
        train_data = EpochWrapper(base_train_loader, 50_000)
        test_data  = EpochWrapper(base_test_loader,  10_000)
        epoch_loss, count = 0.0, 0
        epoch_start = time.time()
        for x, y in tqdm(train_data, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
            params, loss = train_step(params, jnp.array(x), jnp.array(y))
            step_loss = float(jnp.abs(loss))
            epoch_loss += step_loss; count += 1; global_step += 1
            wandb.log({"train/loss": step_loss}, step=global_step)

            if count % 5 == 0:
                train_acc = evaluate(params, train_data)
                val_acc   = evaluate(params, test_data)
                elapsed   = time.time() - start_time
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                wandb.log({"train/accuracy": train_acc}, step=global_step)
                wandb.log({"val/accuracy": val_acc}, step=global_step)
                wandb.log({"elapsed_time_s": elapsed}, step=global_step)
                log(f"Epoch {epoch+1}/{epochs} - Batch {count} - Loss: {step_loss:.4f} - Train: {train_acc:.4f} - Val: {val_acc:.4f}")

        avg_loss   = epoch_loss / count
        epoch_time = time.time() - epoch_start

        loss_history.append(avg_loss)

        wandb.log({"train/epoch_loss": avg_loss}, step=epoch)
        wandb.log({"epoch_time_s": epoch_time}, step=epoch)
        log(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")

    total_time = time.time() - start_time
    wandb.log({"final_val_accuracy": val_acc_history[-1]}, step=0)
    wandb.log({"total_training_time_s": total_time}, step=0)
    wandb.finish()
    return {"name": name, "loss_history": loss_history,
            "train_accuracy_history": train_acc_history, "val_accuracy_history": val_acc_history,
            "final_accuracy": val_acc_history[-1], "time": total_time, "peak_gpu_memory_mb": 0}


def train_pytorch_model(name, epochs=EPOCHS, learning_rate=0.001):
    log(f"--- Training (PyTorch): {name} ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()

    model     = CNN_PyTorch(classes=10).to(device)
    optimizer = toptim.Adam(model.parameters(), lr=learning_rate)
    criterion = tnn.CrossEntropyLoss()

    aim_run = wandb.init(project="pjax", name=name)
    wandb.config.update({
        "dataset": "CIFAR-10", "batch_size": BATCH_SIZE, "epochs": epochs,
        "model": "CNN_PyTorch", "framework": "pytorch",
        "optimizer": "Adam", "learning_rate": learning_rate,
        "widths": PROXY_WIDTHS, "whiten_width": WHITEN_WIDTH, "scaling_factor": SCALING_FACTOR,
        "device": str(device),
    })

    def evaluate(model, loader):
        model.eval()
        accs = []
        with torch.no_grad():
            for x, y in loader:
                x = torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
                y = torch.tensor(y).to(device)
                preds = model(x).argmax(1)
                accs.append((preds == y).float().mean().item())
        model.train()
        return sum(accs) / len(accs) if accs else 0.0

    loss_history, train_acc_history, val_acc_history = [], [], []
    start_time = time.time()
    global_step = 0

    model.train()
    for epoch in range(epochs):
        train_data = EpochWrapper(base_train_loader, 50_000)
        test_data  = EpochWrapper(base_test_loader,  10_000)
        epoch_loss, count = 0.0, 0
        epoch_start = time.time()
        for x, y in tqdm(train_data, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
            x = torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
            y = torch.tensor(y).to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward(); optimizer.step()
            epoch_loss += loss.item(); count += 1; global_step += 1
            wandb.log({"train/loss": loss.item()}, step=global_step)

            if count % 5 == 0:
                train_acc = evaluate(model, train_data)
                val_acc   = evaluate(model, test_data)
                elapsed   = time.time() - start_time
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                wandb.log({"train/accuracy": train_acc}, step=global_step)
                wandb.log({"val/accuracy": val_acc}, step=global_step)
                wandb.log({"elapsed_time_s": elapsed}, step=global_step)
                log(f"Epoch {epoch+1}/{epochs} - Batch {count} - Loss: {loss.item():.4f} - Train: {train_acc:.4f} - Val: {val_acc:.4f}")
                model.train()

        avg_loss   = epoch_loss / count
        epoch_time = time.time() - epoch_start

        loss_history.append(avg_loss)

        wandb.log({"train/epoch_loss": avg_loss}, step=epoch)
        wandb.log({"epoch_time_s": epoch_time}, step=epoch)
        log(f"Epoch {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f}")

    total_time = time.time() - start_time
    wandb.log({"final_val_accuracy": val_acc_history[-1]}, step=0)
    wandb.log({"total_training_time_s": total_time}, step=0)
    wandb.finish()
    return {"name": name, "loss_history": loss_history,
            "train_accuracy_history": train_acc_history, "val_accuracy_history": val_acc_history,
            "final_accuracy": val_acc_history[-1], "time": total_time}


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = [
        train_pjax_model("PJAX CNN (DR)", optim.DouglasRachford),
        train_pytorch_model("PyTorch CNN Baseline"),
    ]

    df = pd.DataFrame(results)
    print("\n--- Final Results ---")
    print(df[["name", "final_accuracy", "time"]])

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))
    for res in results:
        plt.plot(res["val_accuracy_history"], label=f"{res['name']} (Val)", linewidth=2)
    plt.title("Validation Accuracy – Proxy-Net Architecture", fontsize=14)
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    plt.tight_layout()
    plt.savefig("cifar10_results.png")
    log("Saved plot to cifar10_results.png")