import sys, os
sys.path.insert(0, os.path.abspath('.'))
import torch
from frameworks.ptorch import config
from frameworks.ptorch.nn.modules import Linear
torch.manual_seed(42)
X = torch.randn(128, 784, requires_grad=False)
Y = torch.randn(128, 10, requires_grad=False)

config.muon_activations = False
l = Linear(784, 10)
out_f = l(X)
loss_f = torch.nn.functional.mse_loss(out_f, Y)
loss_f.backward()
grad_w_f = l.weight.grad.clone()

l.weight.grad = None
l.bias.grad = None
config.muon_activations = True
out_t = l(X)
loss_t = torch.nn.functional.mse_loss(out_t, Y)
loss_t.backward()
grad_w_t = l.weight.grad.clone()

print("Diff max:", (grad_w_f - grad_w_t).abs().max().item())
