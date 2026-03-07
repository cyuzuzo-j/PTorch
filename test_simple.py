import torch
from frameworks.ptorch.core import ops

A = torch.randn(2, 3)
B = torch.randn(3, 4)
Z = torch.randn(2, 4)
print("Starting matmul_proj...")
A_new, B_new, Z_new, t = ops.matmul_proj(A, B, Z)
print("Finished matmul_proj!")
