import sys
import os
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["PTORCH_AVERAGE_GRADIENTS"] = "1"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))

from frameworks.ptorch.nn.modules import Linear, ReLU
from frameworks.ptorch.optim_static import ProjectionMuon
from frameworks.ptorch.core.ops import MSEProjection, AverageGradient, AverageGradient
from frameworks.ptorch.core.overrides import apply_overrides
import frameworks.ptorch.config as ptorch_config

apply_overrides()
sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)

D = 4
N_STEPS = 500
N_SEEDS = 1
DEPTHS = [2, 32]
BATCH_SIZE = 16
LR = 0.01

class TargetMLP(nn.Module):
    def __init__(self, depth, use_residual=False, d=D):
        super().__init__()
        self.depth = depth
        self.use_residual = use_residual
        self.layers = nn.ModuleList()
        for i in range(depth):
            # We strictly initialize Linear as a base block, and apply the residual topology
            # via `ProjectedAdd` outside the layer. The internal `residual=True` mode in Linear
            # structurally warps the Newton-Schulz qb norms by analyzing B_eff = I - B, which
            # destroys the optimization condition for targets!
            layer = Linear(d, d, bias=True, omega=1.0, residual=False)
            if use_residual:
                with torch.no_grad():
                    layer.weight.mul_(0.0)
                    if layer.bias is not None:
                        layer.bias.mul_(0.0)
            self.layers.append(layer)
            if i < depth - 1:
                self.layers.append(ReLU())
                
    def forward(self, x):
        residual_x = x
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Linear):
                if self.use_residual:
                    # Our ProjectedAdd override captures this automatically and optimally
                    # splits targets.
                    x = residual_x + layer(x)
                    if i < len(self.layers) - 1:
                        x = self.layers[i+1](x)
                    residual_x = x
                else:
                    x = layer(x)
        return x

def run_experiment(device):
    modes = [
        {"name": "No Muon",       "muon_activations": False, "mode": "relative", "scale": False},
        {"name": "Muon Unscaled", "muon_activations": True,  "mode": "unscaled", "scale": False},
        {"name": "Muon Absolute", "muon_activations": True,  "mode": "absolute", "scale": False},
        {"name": "Muon Relative", "muon_activations": True,  "mode": "relative", "scale": False},
    ]

    results = {}
    
    for mode_cfg in modes:
        mode_name = mode_cfg["name"]
        results[mode_name] = {"Standard": {}, "Residual": {}}
        print(f"\n{'='*50}")
        print(f"=== Mode: {mode_name: <32} ===")
        print(f"{'='*50}")
        
        # update global config
        ptorch_config.update("muon_activations", mode_cfg["muon_activations"])
        ptorch_config.update("muon_activations_mode", mode_cfg["mode"])
        ptorch_config.update("muon_activations_scale", mode_cfg["scale"])

        for use_residual in [False, True]:
            arch_name = "Residual" if use_residual else "Standard"
            for depth in DEPTHS:
                final_loss = 0.0
                for seed in range(N_SEEDS):
                    torch.manual_seed(seed)
                    model = TargetMLP(depth=depth, use_residual=use_residual, d=D).to(device)
                    optimizer = ProjectionMuon(model.parameters()) 
                    
                    with ptorch_config.config.projections(True):
                        for step in range(1, N_STEPS + 1):
                            # Ensure we don't blow up without recovery, catch NaNs early to speed up failing modes
                            x_batch = torch.randn(BATCH_SIZE, D, device=device)
                            y_batch = x_batch.clone()
                            
                            model.train()
                            optimizer.zero_grad()
                            
                            preds = model(x_batch)
                            loss = MSEProjection.apply(preds, y_batch)
                            loss.backward()
                            optimizer.step()
                            
                            current_loss = loss.item() / BATCH_SIZE
                            final_loss = current_loss
                            if torch.isnan(torch.tensor(current_loss)) or current_loss > 1e13:
                                break  # stop early on divergence

                results[mode_name][arch_name][depth] = final_loss
                print(f"  {arch_name: <10} | L={depth:<2} | Final Loss: {final_loss:.6f}")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_experiment(device)
