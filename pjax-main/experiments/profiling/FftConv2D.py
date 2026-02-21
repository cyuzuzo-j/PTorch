"""Profile: FftConv2D layer — forward pass + training step."""
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
        self.conv = nn.FftConv2D(in_features_x=16, in_features_y=16,
                                 in_channels=1, out_channels=8,
                                 kernel_shape=3)
        self.relu = nn.ReLU(8)
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

save_graph(loss_fn, params, "fftconv2d_graph.png", "FftConv2D — Computation Graph",
           figsize=(14, 10), font_size=5)

# --- train ---
opt = optim.AlternatingProjections(steps_per_update=10)
params, _ = opt.update(loss_fn, params)

with jax.profiler.trace(os.path.join(OUT, "fftconv2d_trace"), create_perfetto_link=False):
    for _ in range(3):
        params, loss = opt.update(loss_fn, params)
    jnp.array(0.0).block_until_ready()

jax.profiler.save_device_memory_profile(os.path.join(OUT, "fftconv2d_memory.prof"))
print(f"FftConv2D — final loss: {loss:.6f}")
print(f"Trace  → {OUT}/fftconv2d_trace")
print(f"Memory → {OUT}/fftconv2d_memory.prof")
