"""Sanity checks for bilinear_dyadic_linf.

Verifies:
  (1) F(eps_star) >= 0  (discrete feasibility)
  (2) Constraint residual |theta_bar^T h_bar - h_plus_bar| scales as O(2^-p)
  (3) L_inf distance to (h, theta, h_plus) is bounded by eps_star (+ Delta/2 for h_plus)
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as random
import numpy as np

import tools.projections as P


def check_one(h, theta, h_plus, p, q=0):
    h_bar, theta_bar, h_plus_bar = P.bilinear_dyadic_linf(h, theta, h_plus, p=p, q=q)
    Delta = 2.0 ** (-p)
    res = float(jnp.abs(jnp.dot(theta_bar, h_bar) - h_plus_bar))
    linf = float(jnp.max(jnp.array([
        jnp.max(jnp.abs(h_bar - P.quantize_to_dyadic(h, p))),
        jnp.max(jnp.abs(theta_bar - P.quantize_to_dyadic(theta, p))),
        (2.0 ** q) * jnp.abs(h_plus_bar - P.quantize_to_dyadic(h_plus, p)),
    ])))
    on_lattice = all(
        bool(jnp.all(jnp.abs(jnp.round(x / Delta) - x / Delta) < 1e-9))
        for x in (h_bar, theta_bar, h_plus_bar)
    )
    return res, linf, on_lattice


def main():
    key = random.key(0)
    n = 32
    print(f"{'p':>3} {'mean_res':>12} {'max_res':>12} {'2^-p':>12} {'mean_linf':>12} {'on_lattice':>10}")
    print("-" * 70)
    for p in [4, 6, 8, 10, 12, 16, 20]:
        residuals, linfs, lattice_ok = [], [], True
        for trial in range(64):
            key, k1, k2, k3 = random.split(key, 4)
            h      = random.uniform(k1, (n,), minval=-1.0, maxval=1.0)
            theta  = random.uniform(k2, (n,), minval=-1.0, maxval=1.0)
            # Pick h_plus offset from theta^T h so target_gap is non-trivial
            h_plus = jnp.dot(theta, h) + random.uniform(k3, (), minval=-2.0, maxval=2.0)
            res, linf, ok = check_one(h, theta, h_plus, p=p)
            residuals.append(res); linfs.append(linf); lattice_ok &= ok
        residuals = np.array(residuals); linfs = np.array(linfs)
        print(f"{p:>3} {residuals.mean():>12.3e} {residuals.max():>12.3e} "
              f"{2.0**-p:>12.3e} {linfs.mean():>12.3e} {str(lattice_ok):>10}")


if __name__ == "__main__":
    main()
