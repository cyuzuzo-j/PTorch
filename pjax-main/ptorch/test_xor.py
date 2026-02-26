import torch
import torch.nn as nn
from ptorch.nn.modules import LinearBias, ReLU
from ptorch.core.ops import MSEProjection
from ptorch.optim_static import AlternatingProjections

torch.manual_seed(42)
torch.set_default_dtype(torch.float32)

class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        # PyTorch layers using our custom Alternating Projections ops
        self.l1 = LinearBias(2, 4)
        self.r1 = ReLU(4)
        self.l2 = LinearBias(4, 1)

    def forward(self, x):
        x = self.l1(x)
        x = self.r1(x)
        x = self.l2(x)
        return x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing PyTorch Alternating Projections: XOR Classifier on {device}")

    # 1. Dataset (XOR)
    X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], device=device)
    y = torch.tensor([[0.], [1.], [1.], [0.]], device=device)

    # 2. Model & Optimizer
    model = XORNet().to(device)
    optimizer = AlternatingProjections(model.parameters())

    print("Initial Predictions:")
    with torch.no_grad():
        print(model(X))

    # 3. Training Loop
    epochs = 1000
    for epoch in range(epochs):
        # Forward pass
        preds = model(X)
        
        # Loss & Target Projection calculation
        # MSEProjection returns the target value for the preceding nodes, 
        # which acts as the 'gradient' signal during topological descent.
        loss = MSEProjection.apply(preds, y).mean()
        
        # Backward Pass (Topological Descent)
        # Passing y back to the root ensures the graph projects toward the ground truth.
        optimizer.zero_grad()
        
        # In the pjax MSE prox, the projection target for both inputs is `(preds + target) / 2`
        # We need to explicitly pass this target to `backward` since `loss.backward()` 
        # defaults to passing `1.0`.
        target_for_preds = (preds + y) / 2.0
        preds.backward(target_for_preds)
        
        # Optimizer Step (Update parameters to their projected states)
        optimizer.step()
        
        if epoch % 100 == 0:
            with torch.no_grad():
                mse = torch.nn.functional.mse_loss(model(X), y)
                print(f"Epoch {epoch:4d} | MSE: {mse.item():.6f}")

    print("\nFinal Predictions:")
    with torch.no_grad():
        final_preds = model(X)
        for i in range(4):
            print(f"In: {X[i].tolist()} -> Pred: {final_preds[i].item():.4f} (True: {y[i].item()})")

if __name__ == "__main__":
    main()
