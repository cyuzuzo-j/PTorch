import torch
import torch.nn.functional as F

x = torch.sign(torch.randn(1, 256))
w = torch.sign(torch.randn(10, 256))

out = F.linear(x, w)
print("Max absolute logit:", out.abs().max().item())
print("Softmax:", F.softmax(out, dim=-1))
