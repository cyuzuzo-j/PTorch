import torch
import torch.nn.functional as F
from frameworks.ptorch.core import ops

class MockCtx:
    def __init__(self):
        self.saved_tensors = None
    
    def save_for_backward(self, *args):
        self.saved_tensors = args

def run_tests():
    print("Testing forward...")
    predictions = torch.randn(2, 3)
    targets = torch.randn(2, 3)
    ctx = MockCtx()
    loss = ops.MSEProjection.forward(ctx, predictions, targets)
    assert torch.allclose(loss, F.mse_loss(predictions, targets))
    print("Forward passed.")

    print("Testing backward...")
    predictions = torch.randn(2, 3)
    targets = torch.randn(2, 3)
    ctx = MockCtx()
    ops.MSEProjection.forward(ctx, predictions, targets)
    
    projection_prediction, projection_target = ops.MSEProjection.backward(ctx, None)
    
    # Constraint: predictions_new = targets_new
    assert torch.allclose(projection_prediction, projection_target)
    
    # Solution should be the average
    expected_out = (predictions + targets) / 2
    assert torch.allclose(projection_prediction, expected_out)
    assert torch.allclose(projection_target, expected_out)
    print("Backward passed.")

if __name__ == '__main__':
    run_tests()
