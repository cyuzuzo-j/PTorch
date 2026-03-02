"""Profile: MultiHeadAttention layer — forward pass + training step."""
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
        self.attn = nn.MultiHeadAttention(model_features=64,
                                          qkv_features=16,
                                          heads=4)
    def __call__(self, x):
        return self.attn(x)

key = jax.random.PRNGKey(0)
model = Model()
params = model.init(key)

x = pjax.array(jax.random.normal(key, (2, 8, 64)))
target = pjax.array(jax.random.normal(key, (2, 8, 64)))

def loss_fn(params):
    pred = model.apply(params, x)
    return pjax.means_squared_error(pred, target)

save_graph(loss_fn, params, "mha_graph.png", "MultiHeadAttention — Computation Graph",
           figsize=(16, 12), font_size=4)

# --- train ---
opt = optim.AlternatingProjections(steps_per_update=10)
params, _ = opt.update(loss_fn, params)

with jax.profiler.trace(os.path.join(OUT, "mha_trace"), create_perfetto_link=False):
    for _ in range(3):
        params, loss = opt.update(loss_fn, params)
    jnp.array(0.0).block_until_ready()

jax.profiler.save_device_memory_profile(os.path.join(OUT, "mha_memory.prof"))
print(f"MultiHeadAttention — final loss: {loss:.6f}")
print(f"Trace  → {OUT}/mha_trace")
print(f"Memory → {OUT}/mha_memory.prof")
