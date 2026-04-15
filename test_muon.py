import torch
import torch.nn as tnn
from frameworks.ptorch.optim_static import ProjectionMuon

layer = tnn.Linear(16, 8, bias=True)
opt = ProjectionMuon(layer.parameters(), lr=0.01)

x = torch.randn(4, 16)
y = layer(x)
loss = y.sum()
loss.backward()
opt.step()
print("Muon step complete")
