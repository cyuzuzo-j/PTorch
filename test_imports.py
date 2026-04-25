try:
    import torch
    print("torch version:", torch.__version__)
except ImportError:
    print("torch not found")

try:
    import jax
    print("jax version:", jax.__version__)
except ImportError:
    print("jax not found")

try:
    import pandas
    print("pandas version:", pandas.__version__)
except ImportError:
    print("pandas not found")

try:
    import numpy
    print("numpy version:", numpy.__version__)
except ImportError:
    print("numpy not found")
