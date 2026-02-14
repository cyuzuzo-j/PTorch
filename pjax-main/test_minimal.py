#!/usr/bin/env python3
import sys
print("Step 1: Imports starting...", flush=True)

import jax
print("Step 2: JAX imported", flush=True)

import jax.numpy as jnp
print("Step 3: JAX numpy imported", flush=True)

import pjax
print("Step 4: PJAX imported", flush=True)

from pjax import nn
print("Step 5: PJAX nn imported", flush=True)

print("All imports successful!", flush=True)

# Simple test
X = jnp.array([[0.0, 0.0], [1.0, 1.0]])
print(f"Created test array: {X}", flush=True)

print("Test complete!", flush=True)
