import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))

import torch
import torch.nn.functional as F
from experiments.mlp.bench_ptorch import MLP
from ptorch.core.ops import CrossEntropyProjection

def test_mlp_residual():
    device = torch.device("cpu")
    hidden = [128, 128, 128]
    in_features = 784
    classes = 10
    
    model = MLP(hidden, in_features, classes).to(device)
    print("Model Architecture:")
    print(model)
    
    x = torch.randn(8, in_features, device=device)
    y = torch.randint(0, classes, (8,), device=device)
    y_oh = F.one_hot(y, num_classes=classes).float()
    
    print("\nRunning Forward Pass...")
    logits = model(x)
    print("Forward Pass Successful. Shape:", logits.shape)
    
    print("\nRunning Backward Pass (Projection)...")
    projected = CrossEntropyProjection.apply(logits, y_oh)
    proj_loss = projected.sum()
    proj_loss.backward()
    print("Backward Pass Successful.")

if __name__ == "__main__":
    test_mlp_residual()
