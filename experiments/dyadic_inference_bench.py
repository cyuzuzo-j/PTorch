"""
Forward-pass inference benchmark: regular float weights vs dyadic / PTQ int8.

Compares four weight representations on speed, memory, and accuracy fidelity:
  fp32          true IEEE float32 weights + matmul (Precision.HIGHEST, no tensor cores)
  tf32          float32 weights at XLA's default matmul precision (TF32 tensor cores) —
                the realistic float baseline on Ampere+/Hopper
  bf16          bfloat16 weights + matmul, fp32 accumulate
  dyadic-int8   per-tensor power-of-two scale 2^-p; true int8->int32 GEMM,
                rescale by 2^-(p_W+p_x)  (a pow2 shift)
  ptq-int8      standard post-training symmetric per-tensor int8 (scale = max/127);
                true int8->int32 GEMM, rescale by s_W*s_x  (a float multiply)

The int8 variants are W8A8 *dynamic*: weights quantized offline, activations
quantized per-forward from their absmax. Both operands are int8, so the matmul
maps to int8 tensor cores (fast on Hopper; slow on pre-Turing GPUs that lack
int8 tensor cores — the script flags this).

Two regimes:
  * scaled square GEMMs (--sizes) → compute-bound speed story.
  * real MNIST model 784->128->10 → memory + accuracy-fidelity story
    (its forward is launch-bound, so speed there is dispatch noise).

NOTE: this script deliberately does NOT enable jax_enable_x64, so fp32 is a true
4-byte float and the GPU tensor cores engage. (The training script does enable
x64 for solver stability — keep the two separate.)

Run (on a Hopper node):
  python experiments/dyadic_inference_bench.py
  python experiments/dyadic_inference_bench.py --sizes 4096,8192 --no-mnist
"""
import os, sys, json, time, argparse, subprocess
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax  # NB: no jax_enable_x64 on purpose
import jax.numpy as jnp
import jax.random as random
from jax import lax
import numpy as np

import tools.projections as projections

I8_MAX = 127  # symmetric int8 range used for clipping


# ---------------------------------------------------------------- primitives
def leaky(h):
    return jnp.where(h > 0, h, 0.1 * h)


def idot(A, B):
    """int8 (M,K) @ int8 (K,N) -> int32 (M,N) — the tensor-core GEMM."""
    return lax.dot_general(A, B, (((1,), (0,)), ((), ())),
                           preferred_element_type=jnp.int32)


def bdot(A, B):
    """bf16 (M,K) @ bf16 (K,N) -> f32 (M,N), tensor-core path with f32 accumulate."""
    return lax.dot_general(A, B, (((1,), (0,)), ((), ())),
                           preferred_element_type=jnp.float32)


def fdot(A, B, precision):
    """f32 GEMM at a chosen precision. HIGHEST = true IEEE fp32; DEFAULT = TF32
    tensor cores (what XLA picks for f32 matmuls on Ampere+/Hopper by default)."""
    return lax.dot_general(A, B, (((1,), (0,)), ((), ())),
                           preferred_element_type=jnp.float32, precision=precision)


# weight quantizers (offline, one-time)
def quant_dyadic(W, bits=8):
    p = projections.dyadic_scale_pertensor(W, bits)        # pow2 exponent (scalar)
    m = projections.pack_dyadic_dyn(W, p, bits)            # int8 mantissa
    return m, p


def quant_ptq(W, bits=8):
    s = jnp.max(jnp.abs(W)) / (2 ** (bits - 1) - 1)        # free float scale
    m = jnp.clip(jnp.round(W / s), -I8_MAX, I8_MAX).astype(jnp.int8)
    return m, s


# activation quantizers (dynamic, per-forward) — return (mantissa int8, scale-info)
def aquant_dyadic(x, bits=8):
    p = projections.dyadic_scale_pertensor(x, bits)
    return projections.pack_dyadic_dyn(x, p, bits), p


def aquant_ptq(x, bits=8):
    s = jnp.max(jnp.abs(x)) / I8_MAX
    return jnp.clip(jnp.round(x / s), -I8_MAX, I8_MAX).astype(jnp.int8), s


# ---------------------------------------------------------------- forwards
def build_forward(variant, W0, W1, bits=8):
    """Returns (forward(x)->logits_f32, weight_bytes). Weight quant is done here
    (offline); only activation-quant + GEMM + rescale live inside forward()."""
    n_params = int(W0.size + W1.size)

    if variant in ("fp32", "tf32"):
        # fp32 = true IEEE fp32 (no tensor cores); tf32 = XLA's default f32 matmul
        # precision (TF32 tensor cores) — the realistic float baseline on Hopper.
        prec = jax.lax.Precision.HIGHEST if variant == "fp32" else jax.lax.Precision.DEFAULT
        A0, A1 = W0.astype(jnp.float32), W1.astype(jnp.float32)
        def fwd(x):
            return fdot(A1, leaky(fdot(A0, x, prec)), prec)
        return fwd, n_params * 4

    if variant == "bf16":
        A0, A1 = W0.astype(jnp.bfloat16), W1.astype(jnp.bfloat16)
        def fwd(x):
            h = leaky(bdot(A0, x.astype(jnp.bfloat16)))
            return bdot(A1, h.astype(jnp.bfloat16))
        return fwd, n_params * 2

    if variant == "dyadic-int8":
        m0, p0 = quant_dyadic(W0, bits)
        m1, p1 = quant_dyadic(W1, bits)
        def fwd(x):
            mx, px = aquant_dyadic(x, bits)
            h = idot(m0, mx).astype(jnp.float32) * (2.0 ** -(p0 + px))
            h = leaky(h)
            mh, ph = aquant_dyadic(h, bits)
            return idot(m1, mh).astype(jnp.float32) * (2.0 ** -(p1 + ph))
        # int8 mantissas + one f32 exponent per tensor
        return fwd, n_params * (bits // 8) + 2 * 4

    if variant == "ptq-int8":
        m0, s0 = quant_ptq(W0, bits)
        m1, s1 = quant_ptq(W1, bits)
        def fwd(x):
            mx, sx = aquant_ptq(x, bits)
            h = idot(m0, mx).astype(jnp.float32) * (s0 * sx)
            h = leaky(h)
            mh, sh = aquant_ptq(h, bits)
            return idot(m1, mh).astype(jnp.float32) * (s1 * sh)
        return fwd, n_params * (bits // 8) + 2 * 4

    raise ValueError(f"unknown variant {variant!r}")


# ---------------------------------------------------------------- speed
def square_weights(size, batch, key):
    k0, k1, k2 = random.split(key, 3)
    scale = (2.0 / size) ** 0.5
    W0 = random.normal(k0, (size, size), jnp.float32) * scale
    W1 = random.normal(k1, (size, size), jnp.float32) * scale
    x = random.normal(k2, (size, batch), jnp.float32)
    return W0, W1, x


def time_forward(fwd, x, iters, trials):
    """Compute-bound timing: run `iters` forwards inside one jitted fori_loop
    (sum the output so XLA can't DCE the work), median over `trials`."""
    loop = jax.jit(lambda x0: lax.fori_loop(
        0, iters, lambda i, acc: acc + jnp.sum(fwd(x0)), jnp.float32(0.0)))
    loop(x).block_until_ready()  # warmup + compile
    samples = []
    for _ in range(trials):
        t = time.perf_counter()
        loop(x).block_until_ready()
        samples.append((time.perf_counter() - t) / iters)
    return float(np.median(samples))


# ---------------------------------------------------------------- peak memory (subprocess)
def measure_peak_bytes(variant, size, batch):
    """Run one forward in a fresh process and read peak device bytes. Peak is
    monotonic per process, so isolating each config in a subprocess is the only
    way to get a clean per-config number."""
    out = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--mem-worker",
         "--variant", variant, "--size", str(size), "--batch", str(batch)],
        capture_output=True, text=True)
    for line in reversed(out.stdout.strip().splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line).get("peak_bytes")
            except json.JSONDecodeError:
                pass
    return None


def _mem_worker(variant, size, batch):
    W0, W1, x = square_weights(size, batch, random.key(0))
    fwd, _ = build_forward(variant, W0, W1)
    jax.jit(fwd)(x).block_until_ready()
    ms = jax.devices()[0].memory_stats() or {}
    print(json.dumps({"peak_bytes": ms.get("peak_bytes_in_use"),
                      "cur_bytes": ms.get("bytes_in_use")}))


# ---------------------------------------------------------------- MNIST fidelity
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def load_mnist_arrays(data_dir="./dataset", n_test=5000):
    from torchvision.datasets import MNIST
    def to_xy(ds, n):
        xs, ys = [], []
        for i in range(min(n, len(ds))):
            img, lab = ds[i]
            xs.append(np.asarray(img, dtype=np.float32).reshape(-1) / 255.0)
            ys.append(lab)
        x = (np.stack(xs) - MNIST_MEAN) / MNIST_STD
        return x.T, np.asarray(ys)             # x: (784, N)
    train = MNIST(data_dir, train=True, download=True)
    test = MNIST(data_dir, train=False, download=True)
    return to_xy(train, 60000), to_xy(test, n_test)


def train_float_mlp(x_tr, y_tr, hidden=128, steps=1500, bs=128, lr=1e-3, seed=0):
    """Quick float32 Adam-trained MLP to get realistic weights for PTQ fidelity."""
    key = random.key(seed)
    k0, k1 = random.split(key)
    in_dim, n = x_tr.shape
    W0 = random.normal(k0, (hidden, in_dim)) * (2.0 / in_dim) ** 0.5
    W1 = random.normal(k1, (10, hidden)) * (2.0 / hidden) ** 0.5
    m = [jnp.zeros_like(W0), jnp.zeros_like(W1)]
    v = [jnp.zeros_like(W0), jnp.zeros_like(W1)]
    x_tr_j, y_tr_j = jnp.asarray(x_tr), jnp.asarray(y_tr)

    def loss_fn(W0, W1, xb, yb):
        logits = W1 @ leaky(W0 @ xb)               # (10, B)
        logp = logits - jax.scipy.special.logsumexp(logits, axis=0, keepdims=True)
        return -jnp.mean(logp[yb, jnp.arange(yb.shape[0])])

    grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1)))

    @jax.jit
    def adam_step(W0, W1, m, v, t, xb, yb):
        _, (g0, g1) = grad_fn(W0, W1, xb, yb)
        new_m, new_v, new_W = [], [], []
        for W, g, mi, vi in zip((W0, W1), (g0, g1), m, v):
            mi = 0.9 * mi + 0.1 * g
            vi = 0.999 * vi + 0.001 * g * g
            mhat = mi / (1 - 0.9 ** t)
            vhat = vi / (1 - 0.999 ** t)
            new_W.append(W - lr * mhat / (jnp.sqrt(vhat) + 1e-8))
            new_m.append(mi); new_v.append(vi)
        return new_W[0], new_W[1], new_m, new_v

    rng = np.random.default_rng(seed)
    for step in range(1, steps + 1):
        idx = rng.integers(0, n, size=bs)
        xb, yb = x_tr_j[:, idx], y_tr_j[idx]
        W0, W1, m, v = adam_step(W0, W1, m, v, step, xb, yb)
    return W0, W1


def fidelity(variant, W0, W1, x_te, y_te, ref_logits=None):
    fwd, wbytes = build_forward(variant, W0, W1)
    logits = jax.jit(fwd)(jnp.asarray(x_te))
    acc = float(jnp.mean(jnp.argmax(logits, axis=0) == jnp.asarray(y_te)))
    mse = (float(jnp.mean((logits - ref_logits) ** 2))
           if ref_logits is not None else 0.0)
    return acc, mse, wbytes, logits


# ---------------------------------------------------------------- main
def device_note():
    dev = jax.devices()[0]
    kind = getattr(dev, "device_kind", str(dev))
    return dev, kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="fp32,tf32,bf16,dyadic-int8,ptq-int8")
    ap.add_argument("--sizes", default="512,1024,2048,4096,8192",
                    help="square GEMM sizes (in=hidden=out=size, batch=size)")
    ap.add_argument("--batch", type=int, default=0,
                    help="fixed batch for speed (0 = batch equals size)")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--mnist", dest="mnist", action="store_true", default=True)
    ap.add_argument("--no-mnist", dest="mnist", action="store_false")
    ap.add_argument("--mem", dest="mem", action="store_true", default=True,
                    help="measure peak device memory via subprocess workers")
    ap.add_argument("--no-mem", dest="mem", action="store_false")
    ap.add_argument("--mem-sizes", default="",
                    help="square sizes to peak-measure (default: 2 largest)")
    ap.add_argument("--out", default="dyadic_inference_bench")
    # internal subprocess worker:
    ap.add_argument("--mem-worker", action="store_true")
    ap.add_argument("--variant", default="fp32")
    ap.add_argument("--size", type=int, default=4096)
    args = ap.parse_args()

    if args.mem_worker:
        _mem_worker(args.variant, args.size, args.batch or args.size)
        return

    import pandas as pd
    import matplotlib; matplotlib.use("Agg")
    import seaborn as sns
    import matplotlib.pyplot as plt

    dev, kind = device_note()
    variants = [v.strip() for v in args.variants.split(",")]
    sizes = [int(s) for s in args.sizes.split(",")]
    print(f"Device: {kind}   |   variants: {variants}")
    print("(int8 tensor cores: present on Turing+/Hopper; absent on Pascal — "
          "if int8 is SLOWER than fp32 below, this GPU lacks them.)\n")

    rows = []

    # ---- speed sweep (scaled square GEMMs) ----
    print(f"{'size':>6} {'batch':>6} {'variant':>12} {'us/fwd':>10} "
          f"{'TFLOP/s':>9} {'Mspl/s':>8}")
    for size in sizes:
        batch = args.batch or size
        W0, W1, x = square_weights(size, batch, random.key(1))
        flops = 4.0 * size * size * batch          # two GEMMs, 2*M*K*N each
        for v in variants:
            fwd, wbytes = build_forward(v, W0, W1)
            us = time_forward(fwd, x, args.iters, args.trials) * 1e6
            tflops = flops / (us * 1e-6) / 1e12
            mspls = batch / (us * 1e-6) / 1e6
            rows.append(dict(device=kind, section="speed", size=size, batch=batch,
                             variant=v, latency_us=us, tflops=tflops,
                             samples_per_s=batch / (us * 1e-6), weight_bytes=wbytes,
                             peak_bytes=np.nan, test_acc=np.nan, logit_mse=np.nan))
            print(f"{size:>6} {batch:>6} {v:>12} {us:>10.1f} {tflops:>9.1f} {mspls:>8.2f}")

    # ---- peak device memory (subprocess, clean per-config) ----
    if args.mem:
        mem_sizes = ([int(s) for s in args.mem_sizes.split(",")]
                     if args.mem_sizes else sorted(sizes)[-2:])
        print(f"\n[peak device memory @ sizes {mem_sizes} — subprocess per config]")
        for size in mem_sizes:
            batch = args.batch or size
            for v in variants:
                pb = measure_peak_bytes(v, size, batch)
                for r in rows:
                    if r["section"] == "speed" and r["size"] == size and r["variant"] == v:
                        r["peak_bytes"] = pb
                gb = pb / 1e9 if pb else float("nan")
                print(f"  size={size:>6} {v:>12}: peak {gb:6.3f} GB")

    # ---- MNIST fidelity (real model, trained float weights) ----
    if args.mnist:
        print("\n[MNIST 784->128->10 fidelity — training float MLP ...]")
        (x_tr, y_tr), (x_te, y_te) = load_mnist_arrays()
        W0, W1 = train_float_mlp(x_tr, y_tr)
        ref_acc, _, _, ref_logits = fidelity("fp32", W0, W1, x_te, y_te)
        print(f"  float baseline test acc: {ref_acc:.4f}")
        print(f"  {'variant':>12} {'test_acc':>9} {'logit_mse':>11} {'weightKB':>9} {'vs fp32':>8}")
        fp32_bytes = None
        for v in variants:
            acc, mse, wbytes, _ = fidelity(v, W0, W1, x_te, y_te, ref_logits)
            if v == "fp32":
                fp32_bytes = wbytes
            ratio = fp32_bytes / wbytes if fp32_bytes else 1.0
            rows.append(dict(device=kind, section="mnist", size=128, batch=len(y_te),
                             variant=v, latency_us=np.nan, tflops=np.nan,
                             samples_per_s=np.nan, weight_bytes=wbytes,
                             peak_bytes=np.nan, test_acc=acc, logit_mse=mse))
            print(f"  {v:>12} {acc:>9.4f} {mse:>11.4e} {wbytes/1e3:>9.2f} {ratio:>7.1f}x")

    # ---- save ----
    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out}.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2, ax3, ax4 = axes.flat
    pal = dict(zip(variants, sns.color_palette("viridis", len(variants))))
    sp = df[df.section == "speed"]

    for v in variants:
        s = sp[sp.variant == v]
        ax1.plot(s["size"], s["latency_us"], "o-", label=v, color=pal[v])
    ax1.set_xscale("log", base=2); ax1.set_yscale("log")
    ax1.set_xlabel("square size"); ax1.set_ylabel("µs / forward")
    ax1.set_title("Forward latency vs size"); ax1.legend(); ax1.grid(True, alpha=0.2)

    big = sp[sp["size"] == sp["size"].max()].set_index("variant")
    base = big.loc["fp32", "latency_us"] if "fp32" in big.index else big["latency_us"].max()
    speedup = [base / big.loc[v, "latency_us"] if v in big.index else 0 for v in variants]
    ax2.bar(variants, speedup, color=[pal[v] for v in variants])
    ax2.axhline(1.0, ls="--", c="grey")
    ax2.set_ylabel("× vs fp32"); ax2.set_title(f"Speedup at size={int(sp['size'].max())}")
    ax2.tick_params(axis="x", rotation=20)

    if "weight_bytes" in df:
        ref = sp[sp["size"] == sp["size"].max()].set_index("variant")["weight_bytes"]
        ratios = [(ref.get("fp32", ref.max()) / ref.get(v, ref.max())) for v in variants]
        ax3.bar(variants, ratios, color=[pal[v] for v in variants])
        ax3.set_ylabel("× smaller vs fp32"); ax3.set_title("Weight memory reduction")
        ax3.tick_params(axis="x", rotation=20)

    mn = df[df.section == "mnist"]
    if not mn.empty:
        mn = mn.set_index("variant")
        ax4.bar(variants, [mn.loc[v, "test_acc"] if v in mn.index else 0 for v in variants],
                color=[pal[v] for v in variants])
        ax4.set_ylabel("MNIST test acc"); ax4.set_title("Accuracy fidelity (real model)")
        ax4.set_ylim(0, 1); ax4.tick_params(axis="x", rotation=20)
    else:
        ax4.axis("off")

    fig.suptitle(f"Forward pass: float vs dyadic/PTQ int8  —  {kind}")
    fig.tight_layout()
    fig.savefig(f"{args.out}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{args.out}.pdf", dpi=200, bbox_inches="tight")
    print(f"\nSaved: {args.out}.csv, {args.out}.png, {args.out}.pdf")


if __name__ == "__main__":
    main()
