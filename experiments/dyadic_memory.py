"""
Measure actual memory savings (and saturation risk) of packed dyadic storage
on the MNIST shallow-MLP weights/activations from mnist_from_scratch.

For each (p, bits) combination, reports:
  - bytes per element vs float32 baseline
  - fraction of values that saturate at the int range (lossy clipping)
  - max absolute representable value: (2^(bits-1) - 1) * 2^-p
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
import jax.random as random
import numpy as np

import tools.projections as P


def saturation_rate(x, p, bits):
    Delta = 2.0 ** (-p)
    lo = -(1 << (bits - 1))
    hi =  (1 << (bits - 1)) - 1
    m = np.round(np.asarray(x) / Delta)
    sat = np.mean((m < lo) | (m > hi))
    max_repr = hi * Delta
    return float(sat), max_repr


def report_tensor(name, x):
    arr = np.asarray(x)
    print(f"\n{name}: shape={arr.shape} dtype=float32 bytes={arr.nbytes:,}")
    print(f"  range=[{arr.min():.3e}, {arr.max():.3e}]  std={arr.std():.3e}")
    print(f"  {'p':>4} {'bits':>5} {'bytes':>12} {'savings':>9} {'sat_rate':>10} {'max_repr':>12}")
    for bits in (32, 16, 8):
        for p in (4, 6, 8, 10, 12, 14, 16):
            sat, max_repr = saturation_rate(arr, p, bits)
            packed_bytes = arr.size * (bits // 8)
            saving = arr.nbytes / packed_bytes
            mark = "  <- saturates" if sat > 1e-3 else ""
            print(f"  {p:>4} {bits:>5} {packed_bytes:>12,} {saving:>8.1f}x "
                  f"{sat*100:>9.2f}% {max_repr:>12.3e}{mark}")


def main():
    HIDDEN = 128; IN = 784; OUT = 10
    key = random.key(42)
    k0, k1 = random.split(key, 2)
    W0 = random.normal(k0, (HIDDEN, IN)) * 0.01
    W1 = random.normal(k1, (OUT, HIDDEN)) * 0.01

    print("=" * 72)
    print("Memory + saturation report — MNIST shallow MLP at initialization")
    print("(values are pre-training; activations & weights drift during training)")
    print("=" * 72)
    report_tensor("W0  (HIDDEN x IN)",  W0)
    report_tensor("W1  (OUT x HIDDEN)", W1)

    # A representative activation: x_in scaled by MNIST normalization, range ~[-0.42, 2.82]
    rng = np.random.default_rng(0)
    x_in = rng.normal(loc=0.0, scale=1.0, size=(IN, 64)).astype(np.float32)
    report_tensor("x_in (typical activation, std≈1)", jnp.asarray(x_in))

    # Total parameter footprint
    total_float = (W0.size + W1.size) * 4
    print("\n" + "=" * 72)
    print(f"Total weight memory (float32): {total_float:,} bytes "
          f"({total_float/1024:.1f} KiB)")
    for bits in (16, 8):
        packed = (W0.size + W1.size) * (bits // 8)
        print(f"Packed int{bits}:               {packed:,} bytes "
              f"({packed/1024:.1f} KiB)  ⇒ {total_float/packed:.1f}x savings")
    print("=" * 72)
    print("\nPick the smallest (bits, p) where: (a) MNIST training still converges")
    print("(see mnist_dyadic_compare.py) AND (b) saturation rate above is ~0%.")


if __name__ == "__main__":
    main()
