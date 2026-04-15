import sys, os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'frameworks')))
from ptorch.nn.modules import LinearReLU
from ptorch import config

def main():
    print("Testing ReLU Bilinear Projection ...")
    config.update('use_projections', True)
    config.update('projection_norm', 'linf')

    # Basic forward backward test
    torch.manual_seed(42)
    layer = LinearReLU(4, 3, gamma=1.0, norm='linf')
    x = torch.randn(2, 4, requires_grad=True)
    
    y = layer(x)
    print(f"Forward output shape: {y.shape}")
    
    loss = y.sum()
    loss.backward()
    
    print("Backward pass completed.")
    print(f"x.grad is not None: {x.grad is not None}")
    if x.grad is not None:
        print("x.grad:\n", x.grad)
    print(f"layer.weight.grad is not None: {layer.weight.grad is not None}")
    if layer.weight.grad is not None:
        print("layer.weight.grad:\n", layer.weight.grad)

if __name__ == '__main__':
    main()
