import torch
import sys
import os

print("Starting minimal test...", flush=True)

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

print("Importing ptorch modules...", flush=True)
try:
    from frameworks.ptorch.nn import modules as pnn
    from frameworks.ptorch import optim_static as optim
    print("Imports successful!", flush=True)
except Exception as e:
    print(f"Import failed: {e}", flush=True)
    sys.exit(1)

print("Creating a simple Linear layer...", flush=True)
lin = pnn.Linear(10, 5)
print("Linear layer created.", flush=True)

x = torch.randn(2, 10)
print("Input tensor created.", flush=True)

y = lin(x)
print("Forward pass successful!", flush=True)
print(f"Output shape: {y.shape}", flush=True)
