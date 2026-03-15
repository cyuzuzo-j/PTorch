import torch
import torch.nn as nn
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from frameworks.ptorch.nn import modules as pnn
from frameworks.ptorch import optim_static as optim
from frameworks.ptorch.core.ops import MSEProjection, Conversion

class DeepMLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers=32):
        super().__init__()
        layers = []
        # First layer
        layers.append(pnn.Linear(in_features, hidden_features))
        layers.append(pnn.ReLU(hidden_features))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(pnn.Linear(hidden_features, hidden_features))
            layers.append(pnn.ReLU(hidden_features))
            
        # Final layer
        layers.append(pnn.Linear(hidden_features, out_features))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def test_identity():
    # Hyperparameters
    in_features = 8
    hidden_features = 16
    out_features = 8
    num_layers = 3
    batch_size = 32
    num_epochs = 100
    lr = 1.0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)
    
    model = DeepMLP(in_features, hidden_features, out_features, num_layers).to(device)
    optimizer = optim.ProjectionSGD(model.parameters(), lr=lr)
    
    # Simple identity dataset
    x_train = torch.randn(batch_size * 10, in_features).to(device)
    y_train = x_train.clone()
    
    print(f"Starting training for {num_layers}-layer MLP...", flush=True)
    
    for epoch in range(num_epochs):
        model.train()
        
        # Shuffle
        indices = torch.randperm(x_train.size(0))
        x_shuffled = x_train[indices]
        y_shuffled = y_train[indices]
        
        total_loss = 0
        for i in range(0, x_train.size(0), batch_size):
            optimizer.zero_grad()
            
            x_batch = x_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Forward pass
            preds = model(x_batch)
            
            # Use MSEProjection to get targets
            loss = MSEProjection.apply(preds, y_batch)
            
            # Backward pass (propagates targets)
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / (x_train.size(0) / batch_size)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.8f}", flush=True)
            
    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(10, in_features).to(device)
        test_output = model(test_input)
        final_mse = torch.mean((test_input - test_output)**2).item()
        print(f"\nFinal Test MSE: {final_mse:.8f}")
        
    if final_mse < 1e-4:
        print("Success! The deep MLP learned the identity function.")
    else:
        print("Failure. The loss is too high.")

if __name__ == "__main__":
    test_identity()
