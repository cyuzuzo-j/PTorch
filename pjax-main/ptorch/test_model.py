import time
import torch
import torch.nn as nn
from ptorch.nn.modules import Linear, ReLU
from ptorch.core.ops import MSEProjection
from ptorch.optim_static import AlternatingProjections

torch.set_default_dtype(torch.float32)

class MLP(nn.Module):
    def __init__(self, in_features, hidden, out_features):
        super().__init__()
        self.l1 = Linear(in_features, hidden)
        self.r1 = ReLU(hidden)
        self.l2 = Linear(hidden, out_features)
        
    def forward(self, x):
        x = self.l1(x)
        x = self.r1(x)
        x = self.l2(x)
        return x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model and optimizer
    model = MLP(28*28, 256, 10).to(device)
    optimizer = AlternatingProjections(model.parameters())

    # Dummy data
    x = torch.randn(64, 28*28, device=device)
    y_true = torch.randn(64, 10, device=device)

    print("Initial training...")
    for epoch in range(500):
        # 1. Forward Pass
        preds = model(x)
        
        mse = MSEProjection.apply(preds, y_true)

        optimizer.zero_grad()

        mse.backward()

        optimizer.step()
        
        with torch.no_grad():
            mse = torch.nn.functional.mse_loss(model(x), y_true)
            print(f"Epoch {epoch}: MSE = {mse.item():.4f}")

if __name__ == "__main__":
    main()
