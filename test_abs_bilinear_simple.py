import torch
import torch.nn as nn
from frameworks.ptorch.nn.modules import LinearAbs
from frameworks.ptorch import config

def test_linear_abs_forward():
    print("Testing LinearAbs forward pass...")
    config.use_projections = True
    in_features = 10
    out_features = 5
    B = 4
    
    model = LinearAbs(in_features, out_features, bias=True)
    x = torch.randn(B, in_features)
    
    # Forward pass
    y = model(x)
    
    assert y.shape == (B, out_features)
    assert torch.all(y >= 0)  # ABS activation
    print("Forward pass successful.")

def test_linear_abs_backward():
    print("Testing LinearAbs backward pass...")
    config.use_projections = True
    in_features = 8
    out_features = 4
    B = 2
    
    model = LinearAbs(in_features, out_features, bias=True)
    x = torch.randn(B, in_features, requires_grad=True)
    
    y = model(x)
    loss = y.sum()
    loss.backward()
    
    assert x.grad is not None
    assert model.weight.grad is not None
    print("Backward pass successful.")

if __name__ == "__main__":
    try:
        test_linear_abs_forward()
        test_linear_abs_backward()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTests failed: {e}")
        import traceback
        traceback.print_exc()
