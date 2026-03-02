"""Profile: MaxPool2D layer — Conv → ReLU → MaxPool pipeline."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../frameworks")))

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
        self.pool = nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="VALID")
    def __call__(self, x):
        return self.pool(self.relu(self.conv(x)))

key = jax.random.PRNGKey(0)
model = Model()
params = model.init(key)

x = pjax.array(jax.random.normal(key, (4, 16, 16, 1)))
target = pjax.array(jax.random.normal(key, (4, 8, 8, 8)))

def loss_fn(params):
    pred = model.apply(params, x)
    return pjax.means_squared_error(pred, target)

save_graph(loss_fn, params, "maxpool_graph.png", "MaxPool2D — Computation Graph",
           figsize=(14, 10), font_size=5)

# --- train ---
opt = optim.AlternatingProjections(steps_per_update=50)

def train_step(params):
    return opt.update(loss_fn, params)

train_step = jax.jit(train_step)

# Warm up to avoid tracing/compilation in the profile.
for _ in range(5):
    params, _ = train_step(params)
    jnp.array(0.0).block_until_ready()

with jax.profiler.trace(os.path.join(OUT, "maxpool_trace"), create_perfetto_link=False):
    for _ in range(30):
        params, loss = train_step(params)
    jnp.array(0.0).block_until_ready()

jax.profiler.save_device_memory_profile(os.path.join(OUT, "maxpool_memory.prof"))
print(f"MaxPool2D — final loss: {loss:.6}")
print(f"Trace  → {OUT}/maxpool_trace")
print(f"Memory → {OUT}/maxpool_memory.prof")
