"""
Cyclic Projections (hyperbola form) -> MNIST
============================================
Mirrors `from_cyclic_to_ptorch_mnist.py` but routes the bilinear linear-layer
projection through the hyperbola formulation derived from Bauschke-Lal-Wang
Theorems 4.1 / 5.1:

    project (h, W) onto  W h = z   <=>   project (u, v) onto ||u||^2 - ||v||^2 = 2z

solved by 1D Newton on the multiplier lambda in (-1, 1). Equivalent to the
existing `bilinearProj` but written in the hyperbola variables; handles z<0
automatically via the swap symmetry of Theorem 5.1.

Three algorithms (same structure as the original script):
  1. Pure cyclic projections                                  (baseline)
  2. Nonlinear relaxed projections, weights only              (SGD+momentum)
  3. Nonlinear relaxed projections, full (weights + acts)     (SGD+momentum)
"""

import matplotlib
matplotlib.use('Agg')

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
sys.path.append(os.path.abspath(os.getcwd()))

import jax.numpy as jnp
import jax.random as random
import tools.projections as projections
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import jax
import numpy as np
import optax

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# Bilinear matmul projection routed through the hyperbola Newton solve.
matmul_bilinear = projections.matmul_proj_hyperbola
# Input layer (X clamped to data) is a *linear* constraint, so the existing
# exact linear KKT solve is preferred over the bilinear hyperbola one.
matmul_input    = projections.matmul_proj_fixed_X

MNIST_MEAN = 0.1307
MNIST_STD  = 0.3081

NUM_CLASSES   = 10
HIDDEN_WIDTH  = 512
IN_FEATURES   = 784
BATCH_SIZE        = 64
BACKWARDS_PASSES  = 50
NUM_EPOCHS        = 5
DELTA             = 1.0

rand_key = random.key(42)


def load_mnist(batch_size=64, data_dir="./dataset"):
    train_ds = MNIST(data_dir, train=True,  download=True)
    test_ds  = MNIST(data_dir, train=False, download=True)

    def collate_fn(batch):
        images, labels = zip(*batch)
        x = np.array(images, dtype=np.float32) / 255.0
        y = np.array(labels, dtype=np.int64)
        if x.ndim == 3:
            x = x[..., None]
        x = (x - MNIST_MEAN) / MNIST_STD
        return x, y

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, drop_last=False)
    return train_loader, test_loader


@jax.jit
def leakyReluActivationVectorized(x, W, z, slope=0.1):
    H = W @ x
    x_1 = jnp.clip((H + slope * z) / (1.0 + slope * slope), a_max=0)
    y_1 = slope * x_1
    dist_1 = (H - x_1) ** 2 + (z - y_1) ** 2

    x_2 = jnp.clip((H + z) / 2.0, a_min=0)
    y_2 = x_2
    dist_2 = (H - x_2) ** 2 + (z - y_2) ** 2

    new_H = jnp.where(dist_1 < dist_2, x_1, x_2)
    new_z = jnp.where(dist_1 < dist_2, y_1, y_2)
    return new_H, W, new_z


def forward_pass(W0, W1, x_in, y_target):
    x_hidden = W0 @ x_in
    h_hidden = jnp.where(x_hidden > 0, x_hidden, 0.1 * x_hidden)
    x_out    = W1 @ h_hidden
    err_pos = jnp.where(y_target == 1, jnp.where(x_out < DELTA, (x_out - DELTA) ** 2, 0.0), 0.0)
    err_neg = jnp.where(y_target == 0, jnp.where(x_out > 0,     x_out ** 2,           0.0), 0.0)
    return err_pos + err_neg, x_hidden, h_hidden, x_out


def prepare_batch(x_np, y_np):
    batch = x_np.shape[0]
    x_in = jnp.array(x_np.reshape(batch, -1).T, dtype=jnp.float32)
    y_target = jax.nn.one_hot(jnp.array(y_np), NUM_CLASSES).T
    return x_in, y_target


def compute_accuracy(W0, W1, loader, max_batches=None):
    correct, total = 0, 0
    for i, (x_np, y_np) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x_in, y_target = prepare_batch(x_np, y_np)
        _, _, _, x_out = forward_pass(W0, W1, x_in, y_target)
        correct += int(jnp.sum(jnp.argmax(x_out, axis=0) == jnp.array(y_np)))
        total   += len(y_np)
    return correct / total if total > 0 else 0.0


def _eval_step(W0, W1, x_np, y_np, test_iter, test_loader):
    batchError, _, _, x_out_log = forward_pass(W0, W1, *prepare_batch(x_np, y_np))
    current_error = float(jnp.mean(batchError))
    train_acc = float(jnp.mean(jnp.argmax(x_out_log, axis=0) == jnp.array(y_np)))
    try:
        x_val_np, y_val_np = next(test_iter)
    except StopIteration:
        test_iter = iter(test_loader)
        x_val_np, y_val_np = next(test_iter)
    x_val_in, y_val_target = prepare_batch(x_val_np, y_val_np)
    _, _, _, x_val_out = forward_pass(W0, W1, x_val_in, y_val_target)
    val_acc = float(jnp.mean(jnp.argmax(x_val_out, axis=0) == jnp.array(y_val_np)))
    return current_error, train_acc, val_acc, test_iter


# ── 1. Pure Cyclic Projections (hyperbola form) ──────────────────────────────
def run_cyclic_projections(train_loader, test_loader):
    print("\n" + "=" * 60)
    print("  1. Pure Cyclic Projections (hyperbola) on MNIST")
    print("=" * 60)

    key = rand_key
    results = []
    key, k1, k2 = random.split(key, 3)
    W0 = random.normal(k1, (HIDDEN_WIDTH, IN_FEATURES)) * 0.01
    W1 = random.normal(k2, (NUM_CLASSES, HIDDEN_WIDTH)) * 0.01

    global_step = 0
    test_iter = iter(test_loader)
    for epoch in range(NUM_EPOCHS):
        epoch_errors = []
        for batch_idx, (x_np, y_np) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):
            x_in, y_target = prepare_batch(x_np, y_np)
            _, x_hidden, h_hidden, x_out = forward_pass(W0, W1, x_in, y_target)

            for _ in range(BACKWARDS_PASSES):
                x_out, _, _ = projections.classifierOutputVector(x_out, jnp.eye(NUM_CLASSES), y_target)
                h_hidden, W1, x_out  = matmul_bilinear(h_hidden, W1, x_out)
                x_hidden, _, h_hidden = leakyReluActivationVectorized(x_hidden, jnp.eye(HIDDEN_WIDTH), h_hidden)
                x_in, W0, x_hidden    = matmul_input(x_in, W0, x_hidden)

            err, ta, va, test_iter = _eval_step(W0, W1, x_np, y_np, test_iter, test_loader)
            epoch_errors.append(err)
            results.append({"Epoch": epoch, "Step": global_step, "Error": err, "TrainAcc": ta, "ValAcc": va})
            global_step += 1

        print(f"  Epoch {epoch+1} -- Avg Error: {np.mean(epoch_errors):.4e}  Test Acc: {compute_accuracy(W0, W1, test_loader):.4f}")

    print(f"Final Test Accuracy: {compute_accuracy(W0, W1, test_loader):.4f}")
    return pd.DataFrame(results)


# ── 2. Nonlinear Relaxed Projections (weights only) ──────────────────────────
def run_nonlinear_weights(train_loader, test_loader):
    print("\n" + "=" * 60)
    print("  2. Nonlinear Relaxed (hyperbola, weights) on MNIST")
    print("=" * 60)

    optim_W0 = optax.sgd(learning_rate=1.0, momentum=0.9)
    optim_W1 = optax.sgd(learning_rate=1.0, momentum=0.9)

    key = rand_key
    results = []
    key, k1, k2 = random.split(key, 3)
    W0 = random.normal(k1, (HIDDEN_WIDTH, IN_FEATURES)) * 0.01
    W1 = random.normal(k2, (NUM_CLASSES, HIDDEN_WIDTH)) * 0.01
    optim_state_W0 = optim_W0.init(W0)
    optim_state_W1 = optim_W1.init(W1)

    global_step = 0
    test_iter = iter(test_loader)
    for epoch in range(NUM_EPOCHS):
        epoch_errors = []
        for batch_idx, (x_np, y_np) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):
            x_in, y_target = prepare_batch(x_np, y_np)
            _, x_hidden, h_hidden, x_out = forward_pass(W0, W1, x_in, y_target)

            for _ in range(BACKWARDS_PASSES):
                x_out, _, _ = projections.classifierOutputVector(x_out, jnp.eye(NUM_CLASSES), y_target)

                h_hidden, w1_proj, x_out = matmul_bilinear(h_hidden, W1, x_out)
                pseudo_W1 = W1 - w1_proj
                upd, optim_state_W1 = optim_W1.update(pseudo_W1, optim_state_W1, W1)
                W1 = optax.apply_updates(W1, upd)

                x_hidden, _, h_hidden = leakyReluActivationVectorized(x_hidden, jnp.eye(HIDDEN_WIDTH), h_hidden)

                x_in, w0_proj, x_hidden = matmul_input(x_in, W0, x_hidden)
                pseudo_W0 = W0 - w0_proj
                upd, optim_state_W0 = optim_W0.update(pseudo_W0, optim_state_W0, W0)
                W0 = optax.apply_updates(W0, upd)

            err, ta, va, test_iter = _eval_step(W0, W1, x_np, y_np, test_iter, test_loader)
            epoch_errors.append(err)
            results.append({"Epoch": epoch, "Step": global_step, "Error": err, "TrainAcc": ta, "ValAcc": va})
            global_step += 1

        print(f"  Epoch {epoch+1} -- Avg Error: {np.mean(epoch_errors):.4e}  Test Acc: {compute_accuracy(W0, W1, test_loader):.4f}")

    print(f"Final Test Accuracy: {compute_accuracy(W0, W1, test_loader):.4f}")
    return pd.DataFrame(results)


# ── 3. Nonlinear Relaxed Projections (full) ──────────────────────────────────
def run_nonlinear_full(train_loader, test_loader):
    print("\n" + "=" * 60)
    print("  3. Nonlinear Relaxed (hyperbola, full) on MNIST")
    print("=" * 60)

    optim_W0       = optax.sgd(learning_rate=1.0, momentum=0.9)
    optim_W1       = optax.sgd(learning_rate=1.0, momentum=0.9)
    optim_h_hidden = optax.sgd(learning_rate=1.0, momentum=0.9)
    optim_x_hidden = optax.sgd(learning_rate=1.0, momentum=0.9)

    key = rand_key
    results = []
    key, k1, k2 = random.split(key, 3)
    W0 = random.normal(k1, (HIDDEN_WIDTH, IN_FEATURES)) * 0.01
    W1 = random.normal(k2, (NUM_CLASSES, HIDDEN_WIDTH)) * 0.01
    optim_state_W0 = optim_W0.init(W0)
    optim_state_W1 = optim_W1.init(W1)

    global_step = 0
    test_iter = iter(test_loader)
    for epoch in range(NUM_EPOCHS):
        epoch_errors = []
        for batch_idx, (x_np, y_np) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")):
            x_in, y_target = prepare_batch(x_np, y_np)
            _, x_hidden, h_hidden, x_out = forward_pass(W0, W1, x_in, y_target)
            optim_state_x_hidden = optim_x_hidden.init(x_hidden)
            optim_state_h_hidden = optim_h_hidden.init(h_hidden)

            for _ in range(BACKWARDS_PASSES):
                x_out, _, _ = projections.classifierOutputVector(x_out, jnp.eye(NUM_CLASSES), y_target)

                h_hidden_proj, w1_proj, x_out = matmul_bilinear(h_hidden, W1, x_out)
                upd, optim_state_W1 = optim_W1.update(W1 - w1_proj, optim_state_W1, W1)
                W1 = optax.apply_updates(W1, upd)
                upd, optim_state_h_hidden = optim_h_hidden.update(h_hidden - h_hidden_proj, optim_state_h_hidden, h_hidden)
                h_hidden = optax.apply_updates(h_hidden, upd)

                x_hidden_proj, _, h_hidden_proj = leakyReluActivationVectorized(x_hidden, jnp.eye(HIDDEN_WIDTH), h_hidden)
                upd, optim_state_h_hidden = optim_h_hidden.update(h_hidden - h_hidden_proj, optim_state_h_hidden, h_hidden)
                h_hidden = optax.apply_updates(h_hidden, upd)
                upd, optim_state_x_hidden = optim_x_hidden.update(x_hidden - x_hidden_proj, optim_state_x_hidden, x_hidden)
                x_hidden = optax.apply_updates(x_hidden, upd)

                x_in, w0_proj, x_hidden_proj = matmul_input(x_in, W0, x_hidden)
                upd, optim_state_W0 = optim_W0.update(W0 - w0_proj, optim_state_W0, W0)
                W0 = optax.apply_updates(W0, upd)
                upd, optim_state_x_hidden = optim_x_hidden.update(x_hidden - x_hidden_proj, optim_state_x_hidden, x_hidden)
                x_hidden = optax.apply_updates(x_hidden, upd)

            err, ta, va, test_iter = _eval_step(W0, W1, x_np, y_np, test_iter, test_loader)
            epoch_errors.append(err)
            results.append({"Epoch": epoch, "Step": global_step, "Error": err, "TrainAcc": ta, "ValAcc": va})
            global_step += 1

        print(f"  Epoch {epoch+1} -- Avg Error: {np.mean(epoch_errors):.4e}  Test Acc: {compute_accuracy(W0, W1, test_loader):.4f}")

    print(f"Final Test Accuracy: {compute_accuracy(W0, W1, test_loader):.4f}")
    return pd.DataFrame(results)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=int, default=0,
                        help="0=all, 1=cyclic, 2=nlr-weights, 3=nlr-full")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--backward_passes", type=int, default=BACKWARDS_PASSES)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    args = parser.parse_args()

    BATCH_SIZE       = args.batch_size
    BACKWARDS_PASSES = args.backward_passes
    NUM_EPOCHS       = args.epochs

    print(f"Config: batch_size={BATCH_SIZE}, backward_passes={BACKWARDS_PASSES}, epochs={NUM_EPOCHS}")

    train_loader, test_loader = load_mnist(batch_size=BATCH_SIZE)

    all_dfs = {}
    if args.algo in (0, 1):
        all_dfs["cyclic"] = run_cyclic_projections(train_loader, test_loader)
    if args.algo in (0, 2):
        train_loader, test_loader = load_mnist(batch_size=BATCH_SIZE)
        all_dfs["nlr_weights"] = run_nonlinear_weights(train_loader, test_loader)
    if args.algo in (0, 3):
        train_loader, test_loader = load_mnist(batch_size=BATCH_SIZE)
        all_dfs["nlr_full"] = run_nonlinear_full(train_loader, test_loader)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    for label, df in all_dfs.items():
        sns.lineplot(data=df, x="Step", y="Error",    label=label, ax=ax1)
        sns.lineplot(data=df, x="Step", y="TrainAcc", label=label, ax=ax2)
        sns.lineplot(data=df, x="Step", y="ValAcc",   label=label, ax=ax3)
    ax1.set_yscale("log"); ax1.set_title("MNIST -- Training Error (hyperbola)")
    ax2.set_ylim(0, 1.05); ax2.set_title("MNIST -- Train Accuracy (hyperbola)")
    ax3.set_ylim(0, 1.05); ax3.set_title("MNIST -- Validation Accuracy (hyperbola)")
    for ax in (ax1, ax2, ax3):
        ax.grid(True, which="both", ls="-", alpha=0.2)
    fig.tight_layout()
    fig.savefig("mnist_cyclic_projections_hyperbola.png", dpi=150)
    print("Done -- figure saved to mnist_cyclic_projections_hyperbola.png")
