import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from frameworks.ptorch.config import config
from frameworks.ptorch.core.ops import MSEProjection
from frameworks.ptorch.nn.modules_experimental import CausalSelfAttention
from frameworks.ptorch.optim_static import ProjectionSGD, ProjectionMuon

def test_convergence():
    torch.manual_seed(42)
    
    # Generate some dummy data
    B, T, D = 4, 16, 32
    X = torch.randn(B, T, D)
    Y_target = torch.randn(B, T, D)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.csa = CausalSelfAttention(
                dim=D,
                num_heads=4,
                num_kv_heads=2,
                rope_base=10000.0,
                qk_gain_init=1.0,
            )
            
        def forward(self, x):
            return self.csa(x)

    def train_loop(use_proj):
        config.use_projections = use_proj
        torch.manual_seed(42)
        model = Model()
        
        # Use ProjectionSGD for projections, standard SGD for baseline
        if use_proj:
            optimizer = ProjectionMuon(model.parameters(), lr=0.001)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            
        losses = []
        for step in range(1000):
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
    for i in range(0, 1000, 10):
        print(f"{i},{losses_baseline[i]:.4f},{losses_proj[i]:.4f}")

    assert losses_proj[-1] < losses_proj[0] * 0.5, "Projection loss did not converge sufficiently"
    # assert abs(losses_proj[-1] - losses_baseline[-1]) < 0.2, "Projection did not perform comparably to baseline"

if __name__ == "__main__":
    test_convergence()
