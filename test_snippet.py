import torch
from frameworks.ptorch.core import ops

a = torch.tensor([23413.8765, 1234.5678])
b = torch.tensor([28358.24, 8465.2345])
z = torch.tensor([1264684.0])
a_new, b_new, z_new = ops._bilinear_proj_core(a, b, z)
print("a_new @ b_new.T:", a_new @ b_new.T)
print("z_new:", z_new)
