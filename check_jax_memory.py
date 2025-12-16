
import jax
print("JAX Version:", jax.__version__)
try:
    dev = jax.devices()[0]
    print("Device:", dev)
    if hasattr(dev, "memory_stats"):
        print("memory_stats:", dev.memory_stats())
    else:
        print("No memory_stats method found")
except Exception as e:
    print("Error:", e)
