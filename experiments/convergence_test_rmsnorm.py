import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from frameworks.ptorch.config import config
from frameworks.ptorch.core.ops import MatMulProjection, RMSNormProjection, MSEProjection
from frameworks.ptorch.nn.modules_experimental import RMSNorm
from frameworks.ptorch.nn.modules import Linear
from frameworks.ptorch.optim_static import ProjectionSGD

def test_convergence():
    torch.manual_seed(42)
    
    # Generate some dummy data
    N, D_in, D_hidden, D_out = 64, 16, 32, 8
    X = torch.randn(N, D_in)
    Y_target = torch.randn(N, D_out)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = Linear(D_in, D_hidden, bias=False)
            self.norm = RMSNorm()
            self.fc2 = Linear(D_hidden, D_out, bias=False)
            
        def forward(self, x):
            x = self.fc1(x)
            x = self.norm(x)
            x = self.fc2(x)
            return x

    def train_loop(use_proj):
        config.use_projections = use_proj
        torch.manual_seed(42)
        model = Model()
        
        # Use ProjectionSGD for projections, standard SGD for baseline
        if use_proj:
            optimizer = ProjectionSGD(model.parameters(), lr=1.0)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            
        losses = []
        for step in range(100):
            optimizer.zero_grad()
            out = model(X)
            
            if use_proj:
                loss = MSEProjection.apply(out, Y_target)
            else:
                loss = F.mse_loss(out, Y_target)
                
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        return losses

    losses_baseline = train_loop(use_proj=False)
    losses_proj = train_loop(use_proj=True)

    print("Baseline final loss:", losses_baseline[-1])
    print("Projection final loss:", losses_proj[-1])
    
    # Output CSV format to verify
    print("Step,Baseline,Projection")
    for i in range(0, 100, 10):
        print(f"{i},{losses_baseline[i]:.4f},{losses_proj[i]:.4f}")

    assert losses_proj[-1] < losses_proj[0] * 0.5, "Projection loss did not converge sufficiently"
    assert abs(losses_proj[-1] - losses_baseline[-1]) < 0.2, "Projection did not perform comparably to baseline"

if __name__ == "__main__":
    test_convergence()