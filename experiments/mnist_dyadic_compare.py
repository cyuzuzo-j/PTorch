"""
Compare continuous bilinear projections vs dyadic-quantized variants on MNIST.

Mirrors the shallow-MLP setup from mnist_from_scratch.ipynb. For each precision
p in {6, 8, 12, 16, infinity}, runs cyclic projections from the same shared
initialization, logs train/val accuracy and per-batch error, and plots the
result. p=infinity = the original (no quantization) continuous matMul / matMulfixedX.

Run:  python experiments/mnist_dyadic_compare.py [--bps N] [--steps S] [--epochs E]
"""
import os, sys, argparse, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use('Agg')

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import tools.projections as projections


# ---- Problem setup (matches mnist_from_scratch) ----
NUM_CLASSES   = 10
HIDDEN_WIDTH  = 128
IN_FEATURES   = 784
BATCH_SIZE    = 64
DELTA         = 1.0
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def load_mnist(batch_size=64, data_dir="./dataset"):
    def collate_fn(batch):
        images, labels = zip(*batch)
        x = np.array(images, dtype=np.float32) / 255.0
        y = np.array(labels, dtype=np.int64)
        if x.ndim == 3: x = x[..., None]
        x = (x - MNIST_MEAN) / MNIST_STD
        return x, y
    train_ds = MNIST(data_dir, train=True,  download=True)
    test_ds  = MNIST(data_dir, train=False, download=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              collate_fn=collate_fn, drop_last=False)
    return train_loader, test_loader


def prepare_batch(x_np, y_np):
    batch = x_np.shape[0]
    x_in = jnp.array(x_np.reshape(batch, -1).T, dtype=jnp.float32)
    y_target = jax.nn.one_hot(jnp.array(y_np), NUM_CLASSES).T
    return x_in, y_target


def forward_pass(W0, W1, x_in, y_target):
    x_hidden = W0 @ x_in
    h_hidden = jnp.where(x_hidden > 0, x_hidden, 0.1 * x_hidden)
    x_out    = W1 @ h_hidden
    err_pos = jnp.where(y_target == 1, jnp.where(x_out < DELTA, (x_out - DELTA) ** 2, 0.0), 0.0)
    err_neg = jnp.where(y_target == 0, jnp.where(x_out > 0,    x_out ** 2,           0.0), 0.0)
    return err_pos + err_neg, x_hidden, h_hidden, x_out


def compute_accuracy(W0, W1, loader, max_batches=None):
    correct, total = 0, 0
    for i, (x_np, y_np) in enumerate(loader):
        if max_batches is not None and i >= max_batches: break
        x_in, y_target = prepare_batch(x_np, y_np)
        _, _, _, x_out = forward_pass(W0, W1, x_in, y_target)
        correct += int(jnp.sum(jnp.argmax(x_out, axis=0) == jnp.array(y_np)))
        total += len(y_np)
    return correct / total if total > 0 else 0.0


# ---- Projection back-ends parameterized by precision p (None ⇒ continuous) ----

def make_projectors(bits, p):
    """Returns (matMul_fn, matMulfixedX_fn) operating on float tensors.

    For packed runs, the projector encodes inputs to int{bits} mantissas with
    implicit scale 2^-p, runs the joint solver in float, and re-encodes outputs.
    The float<->int boundary is the fixed-point analogue: between calls the
    persistent state (W0, W1) lives in int{bits} memory.
    """
    if bits is None:
        def matMul_fn(X, W, Z):       return projections.matMul(X, W, Z)
        def matMulfixedX_fn(X, W, Z): return projections.matMulfixedX(X, W, Z)
        return matMul_fn, matMulfixedX_fn

    def matMul_fn(X, W, Z):
        Xi = projections.pack_dyadic(X, p=p, bits=bits)
        Wi = projections.pack_dyadic(W, p=p, bits=bits)
        Zi = projections.pack_dyadic(Z, p=p, bits=bits)
        Xo, Wo, Zo = projections.matMul_dyadic_packed(Xi, Wi, Zi, p=p, bits=bits)
        return (projections.unpack_dyadic(Xo, p=p),
                projections.unpack_dyadic(Wo, p=p),
                projections.unpack_dyadic(Zo, p=p))

    def matMulfixedX_fn(X, W, Z):
        Xi = projections.pack_dyadic(X, p=p, bits=bits)
        Wi = projections.pack_dyadic(W, p=p, bits=bits)
        Zi = projections.pack_dyadic(Z, p=p, bits=bits)
        Xo, Wo, Zo = projections.matMulfixedX_dyadic_packed(Xi, Wi, Zi, p=p, bits=bits)
        return (projections.unpack_dyadic(Xo, p=p),
                projections.unpack_dyadic(Wo, p=p),
                projections.unpack_dyadic(Zo, p=p))

    return matMul_fn, matMulfixedX_fn


def _saturation_rate(x, p, bits):
    """Fraction of |m| values that hit the int range when packed."""
    arr = np.asarray(x)
    Delta = 2.0 ** (-p)
    lo = -(1 << (bits - 1))
    hi =  (1 << (bits - 1)) - 1
    m = np.round(arr / Delta)
    return float(np.mean((m < lo) | (m > hi)))


def run(bits, p, bps, batches_per_epoch, num_epochs, key, train_loader, test_loader, label):
    matMul_fn, matMulfixedX_fn = make_projectors(bits, p)

    k1, k2 = random.split(key, 2)
    W0 = random.normal(k1, (HIDDEN_WIDTH, IN_FEATURES)) * 0.01
    W1 = random.normal(k2, (NUM_CLASSES,  HIDDEN_WIDTH)) * 0.01

    # For packed runs, persist W0/W1 as int{bits} between batches and decode
    # at batch entry. This is what realizes the actual memory saving.
    if bits is not None:
        W0_int = projections.pack_dyadic(W0, p=p, bits=bits)
        W1_int = projections.pack_dyadic(W1, p=p, bits=bits)
        W0 = projections.unpack_dyadic(W0_int, p=p)
        W1 = projections.unpack_dyadic(W1_int, p=p)
        weight_bytes_packed = int(W0_int.nbytes + W1_int.nbytes)
    else:
        weight_bytes_packed = None
    weight_bytes_float = int(W0.nbytes + W1.nbytes)

    eye_h = jnp.eye(HIDDEN_WIDTH)
    eye_y = jnp.eye(NUM_CLASSES)

    # Warm-up: JIT-compile each projector once
    x_dummy = jnp.zeros((IN_FEATURES, BATCH_SIZE))
    _, xh0, hh0, xo0 = forward_pass(W0, W1, x_dummy, jnp.zeros((NUM_CLASSES, BATCH_SIZE)))
    _Wh, _, _ = matMul_fn(hh0, W1, xo0);       _Wh.block_until_ready()
    _Xh, _, _ = matMulfixedX_fn(x_dummy, W0, xh0); _Xh.block_until_ready()

    sat_W, sat_x, n_sat = 0.0, 0.0, 0  # accumulated saturation diagnostics

    rows, gstep, cum_time = [], 0, 0.0
    for epoch in range(num_epochs):
        epoch_errs = []
        pbar = tqdm(train_loader, desc=f"[{label}] ep {epoch+1}")
        for batch_idx, (x_np, y_np) in enumerate(pbar):
            if batches_per_epoch and batch_idx >= batches_per_epoch: break
            x_in, y_target = prepare_batch(x_np, y_np)

            # ---- timed projection-cycle region ----
            t0 = time.perf_counter()
            _, x_hidden, h_hidden, x_out = forward_pass(W0, W1, x_in, y_target)
            for _ in range(bps):
                x_out, _, _              = projections.classifierOutput(x_out, eye_y, y_target)
                h_hidden, W1, x_out      = matMul_fn(h_hidden, W1, x_out)
                x_hidden, _, h_hidden    = projections.leakyRelu(x_hidden, eye_h, h_hidden)
                x_in, W0, x_hidden       = matMulfixedX_fn(x_in, W0, x_hidden)
            W0.block_until_ready(); W1.block_until_ready()
            step_time = time.perf_counter() - t0
            cum_time += step_time
            # ---- end timed region ----

            # For packed runs: re-pack persistent weights so storage stays int{bits}
            if bits is not None:
                W0_int = projections.pack_dyadic(W0, p=p, bits=bits)
                W1_int = projections.pack_dyadic(W1, p=p, bits=bits)
                W0 = projections.unpack_dyadic(W0_int, p=p)
                W1 = projections.unpack_dyadic(W1_int, p=p)
                # accumulate saturation rates (cheap; sampled per batch)
                sat_W += _saturation_rate(W0, p, bits) + _saturation_rate(W1, p, bits)
                sat_x += _saturation_rate(x_hidden, p, bits) + _saturation_rate(h_hidden, p, bits)
                n_sat += 2

            err, _, _, x_out_log = forward_pass(W0, W1, x_in, y_target)
            current_err = float(jnp.mean(err))
            train_acc   = float(jnp.mean(jnp.argmax(x_out_log, axis=0) == jnp.array(y_np)))
            val_acc     = compute_accuracy(W0, W1, test_loader, max_batches=8)
            epoch_errs.append(current_err)
            rows.append({"Precision": label, "bits": bits if bits else -1,
                         "p": p if p is not None else -1,
                         "Step": gstep, "Error": current_err,
                         "TrainAcc": train_acc, "ValAcc": val_acc,
                         "StepTime": step_time, "WallTime": cum_time})
            gstep += 1
        if epoch_errs:
            print(f"  [{label}] ep {epoch+1} avg_err={np.mean(epoch_errs):.4e}")
    final_acc = compute_accuracy(W0, W1, test_loader)
    bytes_msg = (f"weights {weight_bytes_packed:,}B (int{bits}) "
                 f"vs {weight_bytes_float:,}B (float32) "
                 f"-> {weight_bytes_float/weight_bytes_packed:.1f}x"
                 if bits is not None else f"weights {weight_bytes_float:,}B (float32)")
    sat_msg = (f"  weight-sat={sat_W/n_sat*100:.2f}%  act-sat={sat_x/n_sat*100:.2f}%"
               if n_sat > 0 else "")
    print(f"[{label}] final test acc: {final_acc:.4f}  |  {bytes_msg}{sat_msg}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bps",    type=int, default=10, help="cyclic backward passes per batch")
    ap.add_argument("--steps",  type=int, default=120, help="batches per epoch (cap)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--out",    type=str, default="mnist_dyadic_compare")
    ap.add_argument("--ps",     type=str, default="32@32,8@16,16@16,inf",
                    help=("comma-separated 'bits@p' configs (e.g. 16@10, 8@4); "
                          "'inf' = continuous float32 baseline"))
    args = ap.parse_args()

    precisions = []  # list of (bits, p, label)
    for tok in args.ps.split(","):
        tok = tok.strip()
        if tok.lower() in ("inf", "infinity", "none", "-1"):
            precisions.append((None, None, "float32 (continuous)"))
        else:
            if "@" not in tok:
                raise ValueError(f"--ps token {tok!r} must be 'bits@p' (e.g. 16@8) or 'inf'")
            bits_s, p_s = tok.split("@")
            bits, p = int(bits_s), int(p_s)
            if bits not in (8, 16, 32):
                raise ValueError(f"bits must be 8/16/32, got {bits}")
            precisions.append((bits, p, f"int{bits} p={p}"))

    train_loader, test_loader = load_mnist(BATCH_SIZE)
    key = random.key(42)
    init_key, _ = random.split(key)

    all_rows = []
    for bits, p, label in precisions:
        all_rows.extend(run(bits, p, args.bps, args.steps, args.epochs,
                            init_key, train_loader, test_loader, label))

    df = pd.DataFrame(all_rows)
    df.to_csv(f"{args.out}.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2, ax3, ax4 = axes.flat
    palette = sns.color_palette("viridis", len(precisions))

    sns.lineplot(data=df, x="Step", y="Error",  hue="Precision", palette=palette, ax=ax1)
    ax1.set_yscale("log"); ax1.set_title("Error vs step (log)"); ax1.grid(True, alpha=0.2)

    sns.lineplot(data=df, x="Step", y="ValAcc", hue="Precision", palette=palette, ax=ax2)
    ax2.set_title("Validation accuracy vs step"); ax2.grid(True, alpha=0.2)

    sns.lineplot(data=df, x="WallTime", y="Error",  hue="Precision", palette=palette, ax=ax3)
    ax3.set_yscale("log"); ax3.set_xlabel("Cumulative wall-clock time (s)")
    ax3.set_title("Error vs wall-clock time (log)"); ax3.grid(True, alpha=0.2)

    sns.lineplot(data=df, x="WallTime", y="ValAcc", hue="Precision", palette=palette, ax=ax4)
    ax4.set_xlabel("Cumulative wall-clock time (s)")
    ax4.set_title("Validation accuracy vs wall-clock time"); ax4.grid(True, alpha=0.2)

    fig.suptitle(f"Dyadic quantization vs continuous projections "
                 f"(bps={args.bps}, batches={args.steps}, epochs={args.epochs})")
    fig.tight_layout()

    # Per-precision mean step time, for the record
    print("\n[mean projection-cycle wall time per batch]")
    for label, sub in df.groupby("Precision", sort=False):
        # Drop the first step which can still include compile/dispatch overhead
        st = sub["StepTime"].iloc[1:].mean() if len(sub) > 1 else sub["StepTime"].mean()
        print(f"  {label:>20s}: {st*1e3:7.2f} ms/batch")
    fig.savefig(f"{args.out}.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(f"{args.out}.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {args.out}.csv, {args.out}.pdf, {args.out}.png")


if __name__ == "__main__":
    main()
