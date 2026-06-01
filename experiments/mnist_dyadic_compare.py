"""
Compare continuous bilinear projections vs dyadic-quantized variants on MNIST.

Mirrors the shallow-MLP setup from mnist_from_scratch.ipynb. Each run stores its
weights in a dyadic format: an int{bits} mantissa m with scale 2^-p, value m·2^-p.

Two schemes (--scale):
  pertensor (default): weight-only. p is chosen per weight matrix from its absmax
      (p = floor(bits-1 - log2(max|W|))), so the matrix fills the int{bits} range
      without saturating; the scale is still a power of two (dyadic), just one
      exponent per tensor. Activations are NOT quantized — they're transient
      (recomputed each batch, never stored), so quantizing them only adds solver
      noise for no memory benefit. This is what makes int8 viable: a single global
      exponent can't span both the ~2.8 inputs and the ~0.01 weights in 8 bits.
  global: original fixed-p dyadic on every tensor (X, W, Z). p must leave enough
      integer headroom for the data — e.g. 16@16 ⇒ range ±0.5 saturates MNIST's
      ~2.82 inputs and never learns; 16@12 ⇒ range ±8 works.

'inf' = the original (no quantization) continuous matMul / matMulfixedX baseline.

Run:  python experiments/mnist_dyadic_compare.py [--scale pertensor|global]
                                                 [--ps 8,16,32,inf] [--bps N] [--steps S]
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

import torch
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

def quantize_weight_pertensor(W, bits):
    """Round-trip W through a per-tensor dyadic int{bits} encoding (the value it
    would have after being stored as a mantissa + per-tensor power-of-two exponent)."""
    pw = projections.dyadic_scale_pertensor(W, bits)
    return projections.unpack_dyadic_dyn(projections.pack_dyadic_dyn(W, pw, bits), pw)


def make_projectors(bits, p, scale="pertensor"):
    """Returns (matMul_fn, matMulfixedX_fn) operating on float tensors.

    scale="pertensor" (default): weight-only quantization. Only the persisted
        weight matrix is encoded — as an int{bits} mantissa with a per-tensor
        power-of-two exponent chosen by absmax — while the transient activations
        (X, Z) stay in float. Adapting the exponent to each weight's magnitude is
        what makes narrow widths (int8) work; keeping activations float avoids
        injecting quantization noise into the iterative solver for no memory gain
        (activations are recomputed each batch, never stored).

    scale="global": original fixed-p dyadic. Every tensor (X, W, Z) is snapped to
        a single shared lattice 2^-p, runs the joint solver in float, and is
        re-encoded. p must leave enough integer headroom or the data saturates.
    """
    if bits is None:
        def matMul_fn(X, W, Z):       return projections.matMul(X, W, Z)
        def matMulfixedX_fn(X, W, Z): return projections.matMulfixedX(X, W, Z)
        return matMul_fn, matMulfixedX_fn

    if scale == "pertensor":
        def matMul_fn(X, W, Z):
            Xo, Wo, Zo = projections.matMul(X, W, Z)
            return Xo, quantize_weight_pertensor(Wo, bits), Zo

        def matMulfixedX_fn(X, W, Z):
            Xo, Wo, Zo = projections.matMulfixedX(X, W, Z)
            return Xo, quantize_weight_pertensor(Wo, bits), Zo

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


def run(bits, p, bps, batches_per_epoch, num_epochs, key, train_loader, test_loader,
        label, scale="pertensor"):
    matMul_fn, matMulfixedX_fn = make_projectors(bits, p, scale)

    def encode_weight(W):
        """Encode→decode one weight matrix under the active scheme.
        Returns (W_float, stored_bytes, exponent_used). For 'pertensor' the
        stored size is the mantissa plus one float32 exponent per tensor."""
        if scale == "pertensor":
            pw = projections.dyadic_scale_pertensor(W, bits)
            Wi = projections.pack_dyadic_dyn(W, pw, bits)
            return projections.unpack_dyadic_dyn(Wi, pw), int(Wi.nbytes) + 4, float(pw)
        Wi = projections.pack_dyadic(W, p=p, bits=bits)
        return projections.unpack_dyadic(Wi, p=p), int(Wi.nbytes), p

    k1, k2 = random.split(key, 2)
    W0 = random.normal(k1, (HIDDEN_WIDTH, IN_FEATURES)) * 0.01
    W1 = random.normal(k2, (NUM_CLASSES,  HIDDEN_WIDTH)) * 0.01

    # For packed runs, persist W0/W1 as int{bits} between batches and decode
    # at batch entry. This is what realizes the actual memory saving.
    if bits is not None:
        W0, b0, _ = encode_weight(W0)
        W1, b1, _ = encode_weight(W1)
        weight_bytes_packed = b0 + b1
    else:
        weight_bytes_packed = None
    # float32-equivalent baseline (the run itself uses float64 for solver stability
    # when jax_enable_x64 is on, so count elements × 4 rather than .nbytes).
    weight_bytes_float = int((W0.size + W1.size) * 4)

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
                W0, _, p0 = encode_weight(W0)
                W1, _, p1 = encode_weight(W1)
                # accumulate saturation rates (cheap; sampled per batch)
                sat_W += _saturation_rate(W0, p0, bits) + _saturation_rate(W1, p1, bits)
                if scale == "global":  # activations only quantized in global mode
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
    ap.add_argument("--scale",  type=str, default="pertensor", choices=("pertensor", "global"),
                    help=("'pertensor' (default): weight-only dyadic, per-tensor exponent "
                          "chosen by absmax, activations stay float. 'global': original "
                          "fixed-p dyadic on every tensor (then each --ps token needs '@p')."))
    ap.add_argument("--ps",     type=str, default="8,16,32,inf",
                    help=("comma-separated configs: bare 'bits' (per-tensor exponent is "
                          "auto-derived) or 'bits@p' for a fixed exponent (required under "
                          "--scale global). 'inf' = continuous float32 baseline."))
    args = ap.parse_args()

    precisions = []  # list of (bits, p, label)
    for tok in args.ps.split(","):
        tok = tok.strip()
        if tok.lower() in ("inf", "infinity", "none", "-1"):
            precisions.append((None, None, "float32 (continuous)"))
            continue
        if "@" in tok:
            bits_s, p_s = tok.split("@")
            bits, p = int(bits_s), int(p_s)
        else:
            bits, p = int(tok), None
        if bits not in (8, 16, 32):
            raise ValueError(f"bits must be 8/16/32, got {bits}")
        if args.scale == "global" and p is None:
            raise ValueError(f"--scale global needs an explicit exponent: '{bits}@<p>'")
        label = f"int{bits}" if args.scale == "pertensor" else f"int{bits} p={p}"
        precisions.append((bits, p, label))

    torch.manual_seed(0)  # fix DataLoader shuffle order so runs are reproducible
    train_loader, test_loader = load_mnist(BATCH_SIZE)
    key = random.key(42)
    init_key, _ = random.split(key)

    all_rows = []
    for bits, p, label in precisions:
        all_rows.extend(run(bits, p, args.bps, args.steps, args.epochs,
                            init_key, train_loader, test_loader, label, scale=args.scale))

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

    fig.suptitle(f"Dyadic quantization ({args.scale}) vs continuous projections "
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
