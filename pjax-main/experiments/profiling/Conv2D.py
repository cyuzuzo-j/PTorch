"""Profile: Conv2D (patch-based) layer — forward pass + training step."""
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
        self.conv = nn.Conv2D(in_channels=1, out_channels=8,
                              kernel_shape=(3, 3), strides=(1, 1), padding="SAME")
        self.relu = nn.ReLU_NB()
    def __call__(self, x):
        return self.relu(self.conv(x))

key = jax.random.PRNGKey(0)
model = Model()
params = model.init(key)

x = pjax.array(jax.random.normal(key, (4, 16, 16, 1)))
target = pjax.array(jax.random.normal(key, (4, 16, 16, 8)))

def loss_fn(params):
    pred = model.apply(params, x)
    return pjax.means_squared_error(pred, target)

save_graph(loss_fn, params, "conv2d_graph.png", "Conv2D — Computation Graph")

# --- train ---
opt = optim.AlternatingProjections(steps_per_update=50)
params, _ = opt.update(loss_fn, params)

with jax.profiler.trace(os.path.join(OUT, "conv2d_trace"), create_perfetto_link=False):
    for _ in range(3):
        params, loss = opt.update(loss_fn, params)
    jnp.array(0.0).block_until_ready()

jax.profiler.save_device_memory_profile(os.path.join(OUT, "conv2d_memory.prof"))
print(f"Conv2D — final loss: {loss:.6f}")
print(f"Trace  → {OUT}/conv2d_trace")
print(f"Memory → {OUT}/conv2d_memory.prof")
