"""
Test file confirming the functioning of mse_proj
"""
import torch
import torch.nn.functional as F
from .. import ops

class MockCtx:
    def __init__(self):
        self.saved_tensors = None
    
    def save_for_backward(self, *args):
        self.saved_tensors = args

def test_forward_working():
    predictions = torch.randn(2, 3)
    targets = torch.randn(2, 3)
    ctx = MockCtx()
    loss = ops.MSEProjection.forward(ctx, predictions, targets)
    assert torch.allclose(loss, F.mse_loss(predictions, targets))

def test_backward_working():
    predictions = torch.randn(2, 3)
    targets = torch.randn(2, 3)

    ctx = MockCtx()
    ops.MSEProjection.forward(ctx, predictions, targets)
    
    projection_prediction, projection_target = ops.MSEProjection.backward(ctx, None)
    
    assert torch.allclose(projection_prediction, projection_target)
    
    expected_out = (predictions + targets) / 2
    assert torch.allclose(projection_prediction, expected_out)
    assert torch.allclose(projection_target, expected_out)