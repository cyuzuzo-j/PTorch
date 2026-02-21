import os
import sys
import time
import tracemalloc
import gc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import jax
import jax.numpy as jnp
import torch
import torch.nn as tnn
import torch.optim as toptim
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import pjax
from pjax import nn, optim
from experiments.shared.data import InfiniteCifarLoader

# Configure logging
def log(msg):
    print(f"[INFO] {msg}")

# --- Constants ---
BATCH_SIZE = 128
EPOCHS = 3
IMAGE_SIZE = 32
INPUT_CHANNELS = 3

# Load Data
log("Loading CIFAR-10 dataset using InfiniteCifarLoader...")

hyp = {
    'aug': {
        'flip': True,
        'translate': 4,
        'cutout': 12,
    }
}

base_train_loader = InfiniteCifarLoader('./dataset', train=True, batch_size=BATCH_SIZE, aug=hyp['aug'])
base_test_loader = InfiniteCifarLoader('./dataset', train=False, batch_size=BATCH_SIZE)

class EpochWrapper:
    def __init__(self, loader, num_samples):
        self.loader_iter = iter(loader)
        self.steps = num_samples // loader.batch_size

    def __iter__(self):
        for _ in range(self.steps):
            _, images, labels = next(self.loader_iter)
            images = images.float().permute(0, 2, 3, 1).cpu().numpy()
            labels = labels.cpu().numpy()
            yield images, labels

train_data = EpochWrapper(base_train_loader, 50000)
test_data = EpochWrapper(base_test_loader, 10000)
gc.collect()

# --- Model Definitions ---
class LeNet_FFT(nn.Module):
    """
    LeNet-like architecture implemented with FFT convolution layers for CIFAR10.
    """
    def __init__(self, classes, image_size=IMAGE_SIZE, input_channels=INPUT_CHANNELS):
        super().__init__()
        
        # Conv1: 5x5 valid -> Pool 2x2
        out_size1 = (image_size - 4) // 2
        self.conv1 = nn.FftConv2D(image_size, image_size, input_channels, 4, 5)
        self.relu1 = nn.ReLU(4)
        self.pool1 = nn.MaxPool2D()

        # Conv2: 5x5 valid -> Pool 2x2
        out_size2 = (out_size1 - 4) // 2
        self.conv2 = nn.FftConv2D(out_size1, out_size1, 4, 6, 5)
        self.relu2 = nn.ReLU(6)
        self.pool2 = nn.MaxPool2D()

        self.fc1 = nn.Linear(6 * out_size2 * out_size2, 120)
        self.relu3 = nn.ReLU(120)
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU(84)
        self.out = nn.Linear(84, classes)

    def __call__(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = pjax.reshape(x, (x.shape[0], -1))
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        return self.out(x)

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_shape=(3, 3), padding="SAME")

    def __call__(self, x):
        return self.conv(x)

class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = Conv(channels_in, channels_out)
        self.pool = nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.conv2 = Conv(channels_out, channels_out)
        self.activ = nn.ReLU(channels_out)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.activ(x)
        x = self.conv2(x)
        x = self.activ(x)
        return x

class CNN_PJAX_Standard(nn.Module):
    """
    PJAX CNN model using Standard Convolution.
    """
    def __init__(self, classes=10, image_size=IMAGE_SIZE, input_channels=INPUT_CHANNELS):
        super().__init__()
        widths = dict(block1=64, block2=256, block3=256)
        whiten_kernel_size = 2
        whiten_width = 2 * 3 * whiten_kernel_size**2
        
        self.whiten = nn.Conv2D(
            3, whiten_width, kernel_shape=(whiten_kernel_size, whiten_kernel_size), padding="VALID"
        )
        self.group1 = ConvGroup(whiten_width, widths["block1"])
        self.group2 = ConvGroup(widths["block1"], widths["block2"])
        self.group3 = ConvGroup(widths["block2"], widths["block3"])

        self.pool = nn.MaxPool2D(pool_size=(3, 3), strides=(3, 3))
        print(widths["block3"])
        self.head = nn.Linear(3*3*widths["block3"], classes)

    def __call__(self, x):
        print("before whiten",x.shape)
        x = self.whiten(x)
        print("before group1",x.shape)
        x = self.group1(x)
        print("before group2",x.shape)
        x = self.group2(x)
        print("before group3",x.shape)
        x = self.group3(x)
        print("before pool",x.shape)
        x = self.pool(x)
        print("before reshape",x.shape)
        x = pjax.reshape(x, (x.shape[0], -1))
        print(x.shape)
        return self.head(x)

class CNN_PyTorch(tnn.Module):
    """
    PyTorch Baseline CNN. Matches PJAX_Standard architecture exactly.
    """
    def __init__(self, classes, image_size=IMAGE_SIZE, input_channels=INPUT_CHANNELS):
        super().__init__()
        self.conv1 = tnn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.relu1 = tnn.ReLU()
        self.conv2 = tnn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = tnn.ReLU()
        self.fc = tnn.Linear(32 * image_size * image_size, classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

# --- Training Functions ---

def train_pjax_model(model_class, name, optimizer_cls, epochs=EPOCHS):
    log(f"--- Starting Training: {name} ---")
    key = jax.random.key(0)
    model = model_class(classes=10)
    params = model.init(key)

    optimizer = optimizer_cls(steps_per_update=10)

    @jax.jit
    def train_step(params, x, y):
        def apply_fn(params):
            logits = model.apply(params, x)
            y_one_hot = jax.nn.one_hot(y, num_classes=10).astype(jax.numpy.complex64)
            return pjax.cross_entropy(logits, y_one_hot)
        updated_params, loss = optimizer.update(apply_fn, params)
        return updated_params, loss

    @jax.jit
    def eval_step(params, x, y):
        logits = model.apply(params, x)
        predictions = jnp.argmax(logits.real if jnp.iscomplexobj(logits) else logits, axis=-1)
        return jnp.mean(predictions == y)

    def evaluate(params, loader):
        total_acc, num_batches = 0, 0
        for x, y in loader:
            total_acc += float(eval_step(params, jnp.array(x), jnp.array(y)))
            num_batches += 1
        return total_acc / num_batches if num_batches > 0 else 0

    loss_history, train_acc_history, val_acc_history = [], [], []
    tracemalloc.start()
    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss, count = 0, 0
        for x, y in train_data:
            params, loss = train_step(params, jnp.array(x), jnp.array(y))
            epoch_loss += float(jnp.abs(loss))
            count += 1

        avg_loss = epoch_loss / count
        train_acc = evaluate(params, train_data)
        val_acc = evaluate(params, test_data)
        
        loss_history.append(avg_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        
        log(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

    end_time = time.time()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "name": name,
        "loss_history": loss_history,
        "train_accuracy_history": train_acc_history,
        "val_accuracy_history": val_acc_history,
        "final_accuracy": val_acc_history[-1],
        "time": end_time - start_time,
        "peak_memory_mb": peak / 10**6,
        "peak_gpu_memory_mb": 0
    }


def train_pytorch_model(name, epochs=EPOCHS, learning_rate=0.001):
    log(f"--- Starting Training: {name} ---")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_PyTorch(classes=10).to(device)
    optimizer = toptim.Adam(model.parameters(), lr=learning_rate)
    criterion = tnn.CrossEntropyLoss()

    def evaluate(model, loader):
        model.eval()
        total_acc, num_batches = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x = torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
                y = torch.tensor(y).to(device)
                outputs = model(x)
                _, predicted = torch.max(outputs.data, 1)
                total_acc += (predicted == y).sum().item() / y.size(0)
                num_batches += 1
        model.train()
        return total_acc / num_batches if num_batches > 0 else 0

    loss_history, train_acc_history, val_acc_history = [], [], []
    tracemalloc.start()
    start_time = time.time()

    model.train()
    for epoch in range(epochs):
        epoch_loss, count = 0, 0
        for x, y in train_data:
            x = torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
            y = torch.tensor(y).to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            count += 1

        avg_loss = epoch_loss / count
        train_acc = evaluate(model, train_data)
        val_acc = evaluate(model, test_data)
        
        loss_history.append(avg_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        
        log(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f} - Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")

    end_time = time.time()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0

    return {
        "name": name,
        "loss_history": loss_history,
        "train_accuracy_history": train_acc_history,
        "val_accuracy_history": val_acc_history,
        "final_accuracy": val_acc_history[-1],
        "time": end_time - start_time,
        "peak_memory_mb": peak / 10**6,
        "peak_gpu_memory_mb": peak_gpu_mb
    }

# --- Run Experiments ---
if __name__ == "__main__":
    results = []
    
    # 1. PJAX FFT Momentum
    #results.append(train_pjax_model(LeNet_FFT, "PJAX FFT CNN momentum", optim.AlternatingProjectionsMonumentum))
    # 2. PJAX Standard Momentum
    results.append(train_pjax_model(CNN_PJAX_Standard, "PJAX CNN momentum", optim.AlternatingProjectionsMonumentum))
    # 3. PyTorch Baseline
    results.append(train_pytorch_model("PyTorch CNN Baseline"))

    # --- Metrics Export ---
    df_results = pd.DataFrame(results)
    print("\n--- Final Results ---")
    print(df_results[["name", "final_accuracy", "time", "peak_memory_mb", "peak_gpu_memory_mb"]])
    
    # Optional: Save visualization instead of blocking with plt.show()
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5))
    for res in results:
        plt.plot(res["val_accuracy_history"], label=f"{res['name']} (Val)", linewidth=2)
    plt.title("Validation Accuracy Comparison", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("cifar10_results.png")
    log("Saved validation accuracy plot to cifar10_results.png")