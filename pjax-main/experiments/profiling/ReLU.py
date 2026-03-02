"""Profile: ReLU layer — forward pass + training step."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import jax
from jax import numpy as jnp
import pjax
from pjax import nn, optim
from helpers import save_graph, OUT

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 64)
        self.relu = nn.ReLU(64)
        self.out = nn.Linear(64, 10)
    def __call__(self, x):
        return self.out(self.relu(self.linear(x)))

key = jax.random.PRNGKey(0)
model = Model()
params = model.init(key)

x = pjax.array(jax.random.normal(key, (16, 128)))
target = pjax.array(jax.random.normal(key, (16, 10)))

def loss_fn(params):
    pred = model.apply(params, x)
    return pjax.means_squared_error(pred, target)

save_graph(loss_fn, params, "relu_graph.png", "ReLU — Computation Graph")

# --- train ---
opt = optim.AlternatingProjections(steps_per_update=20)
params, _ = opt.update(loss_fn, params)

with jax.profiler.trace(os.path.join(OUT, "relu_trace"), create_perfetto_link=False):
    for _ in range(3):
        params, loss = opt.update(loss_fn, params)
    jnp.array(0.0).block_until_ready()

jax.profiler.save_device_memory_profile(os.path.join(OUT, "relu_memory.prof"))
print(f"ReLU — final loss: {loss:.6f}")
print(f"Trace  → {OUT}/relu_trace")
print(f"Memory → {OUT}/relu_memory.prof")
