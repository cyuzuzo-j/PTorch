import torch
import os
from frameworks.ptorch.config import config
 
print("--- Testing default PyTorch (sum of gradients) ---")
os.environ["PTORCH_AVERAGE_GRADIENTS"] = "0"
x = torch.tensor([2.0], requires_grad=True)

# 3 branches of x: y = x + 2*x + 3*x
y1 = x 
y2 = 2 * x
y3 = 3 * x
out = y1 + y2 + y3 # out = 6*x, grad should be 6
out.backward()

print(f"Default sum grad (expect 6.0): {x.grad.item()}")

print("\n--- Testing Averaged Gradients ---")
os.environ["PTORCH_AVERAGE_GRADIENTS"] = "1"
x_avg = torch.tensor([2.0], requires_grad=True)

y1_avg = x_avg
y2_avg = 2 * x_avg
y3_avg = 3 * x_avg
out_avg = y1_avg + y2_avg + y3_avg
out_avg.backward()

# grad 1 from y1, 2 from y2, 3 from y3 -> sum is 6 -> averaged is 6/3 = 2!
print(f"Averaged sum grad (expect 2.0 = 6/3): {x_avg.grad.item()}")

print("\n--- Testing ptorch config toggle ---")
config.average_gradients = True
x_cfg = torch.tensor([2.0], requires_grad=True)

out_cfg = x_cfg + 2 * x_cfg + 3 * x_cfg
out_cfg.backward()

print(f"Config toggled grad (expect 2.0): {x_cfg.grad.item()}")