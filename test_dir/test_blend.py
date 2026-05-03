import torch
from frameworks.ptorch.core.ops import process_activation_target, zeropower_via_newtonschulz5
from frameworks.ptorch import config

config.muon_activations = True
config.muon_activations_mode = 'relative'

A = torch.randn(256, 10)
A_proj = A - torch.randn(256, 10) * 0.1

g = A - A_proj

A_new = process_activation_target(A, A_proj)
print(A_new.shape)
