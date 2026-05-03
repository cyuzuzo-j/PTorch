import jax, jax.numpy as jnp
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import sys
from experiments.mlp.bench_memory import _run_in_subprocess

def run():
    print("starting")
    d = {}
    try:
        # width, depth, batch, in_features
        _run_in_subprocess('pjax', 500, 1, 8, 2500, d)
        print("done", d)
    except Exception as e:
        print("error", e)

if __name__ == "__main__":
    run()
