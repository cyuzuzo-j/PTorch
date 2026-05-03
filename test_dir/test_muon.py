import torch
from frameworks.ptorch.core.ops import zeropower_via_newtonschulz5

G = torch.zeros(256, 1000)
# Assume 10 classes. Samples 0-24 are class 0, 25-49 are class 1, etc.
# Their ideal gradients are highly correlated per class.
for c in range(10):
    G[c*25:(c+1)*25, :] = torch.randn(1, 1000) + 0.1 * torch.randn(25, 1000)

G_muon = zeropower_via_newtonschulz5(G)

# Check correlation between sample 0 and sample 1 (same class)
print("Original dot product:", torch.dot(G[0], G[1]).item())
print("Muon dot product:", torch.dot(G_muon[0], G_muon[1]).item())
