"""Profile: Embedding layer — forward pass + training step."""
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
        self.embed = nn.Embedding(num_embeddings=256, features=32)
        self.linear = nn.Linear(32, 10)
    def __call__(self, x):
        return self.linear(self.embed(x))

key = jax.random.PRNGKey(0)
model = Model()
params = model.init(key)

x = pjax.array(jax.random.randint(key, (16, 8), 0, 256))
target = pjax.array(jax.random.normal(key, (16, 8, 10)))

def loss_fn(params):
    pred = model.apply(params, x)
    return pjax.means_squared_error(pred, target)

save_graph(loss_fn, params, "embedding_graph.png", "Embedding — Computation Graph")

# --- train ---
opt = optim.AlternatingProjections(steps_per_update=20)
params, _ = opt.update(loss_fn, params)

with jax.profiler.trace(os.path.join(OUT, "embedding_trace"), create_perfetto_link=False):
    for _ in range(3):
        params, loss = opt.update(loss_fn, params)
    jnp.array(0.0).block_until_ready()

jax.profiler.save_device_memory_profile(os.path.join(OUT, "embedding_memory.prof"))
print(f"Embedding — final loss: {loss:.6f}")
print(f"Trace  → {OUT}/embedding_trace")
print(f"Memory → {OUT}/embedding_memory.prof")
