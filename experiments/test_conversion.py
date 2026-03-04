"""
Test: Conversion layer bridge between neural (gradient-based) and 
projection-based layers.

Architecture:
  Input → nn.Linear (gradient-based, layer 1)
        → Conversion (bridge)
        → LinearBias (projection-based, layer 2) ← FROZEN
        → Output

The output layer is FROZEN so that ONLY the neural input layer
can learn. If the loss decreases, it proves the Conversion layer
is correctly converting projection targets into gradients.

Task: Simple regression on synthetic data (y = sin(x1) + cos(x2)).
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../frameworks')))

import torch
import torch.nn as tnn
import torch.nn.functional as F
from ptorch.nn.modules import LinearBias, ReLU, Conversion
from ptorch.core.ops import MSEProjection
import ptorch.optim_static as ptorch_optim

torch.manual_seed(42)

# ── Synthetic dataset ────────────────────────────
def make_data(n=1000):
    """Create synthetic regression data: y = sin(x1) + cos(x2) + noise."""
    X = torch.randn(n, 4)
    y = torch.sin(X[:, 0]) + torch.cos(X[:, 1]) + 0.1 * torch.randn(n)
    return X, y.unsqueeze(-1)

X_train, y_train = make_data(2000)
X_test, y_test   = make_data(500)


# ── Hybrid model ─────────────────────────────────
class HybridNet(tnn.Module):
    """2-layer network: neural input layer + projection-based output layer.
    
    Layer 1: Standard nn.Linear (trained with gradients via Adam)
    Bridge:  Conversion layer (converts projection targets → gradients)
    Layer 2: LinearBias (FROZEN — projection-based, not updated)
    """
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        # Layer 1: standard gradient-based linear
        self.neural_layer = tnn.Linear(in_features, hidden_features)
        self.activation = tnn.ReLU()
        
        # Bridge: projection targets → gradients
        self.conversion = Conversion()
        
        # Layer 2: projection-based linear (will be FROZEN)
        self.proj_layer = LinearBias(hidden_features, out_features)
    
    def forward(self, x):
        # Neural part (gradient-based)
        x = self.neural_layer(x)
        x = self.activation(x)
        
        # Bridge
        x = self.conversion(x)
        
        # Projection part (frozen)
        x = self.proj_layer(x)
        return x


def train_epoch(model, X, y, neural_optimizer, proj_optimizer, batch_size=64):
    """Train one epoch, return average loss."""
    model.train()
    perm = torch.randperm(X.size(0))
    X_s, y_s = X[perm], y[perm]
    total_loss, n = 0.0, 0
    
    for i in range(0, X.size(0), batch_size):
        xb, yb = X_s[i:i+batch_size], y_s[i:i+batch_size]
        pred = model(xb)
        projected = MSEProjection.apply(pred, yb)
        loss_val = F.mse_loss(pred.detach(), yb)
        
        neural_optimizer.zero_grad()
        if proj_optimizer:
            proj_optimizer.zero_grad()
        projected.sum().backward()
        neural_optimizer.step()
        if proj_optimizer:
            proj_optimizer.step()
        
        total_loss += loss_val.item()
        n += 1
    return total_loss / n


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        return F.mse_loss(model(X), y).item()


# ── Build model ──────────────────────────────────
model = HybridNet(in_features=4, hidden_features=32, out_features=1)

# Save initial weights for later comparison
init_neural_w = model.neural_layer.weight.data.clone()
init_proj_w = model.proj_layer.weight.data.clone()

neural_params = list(model.neural_layer.parameters())
proj_params = list(model.proj_layer.parameters())

n_epochs = 50


# ══════════════════════════════════════════════════
# PHASE 1: Output layer FROZEN — only neural layer trains
# ══════════════════════════════════════════════════
print("=" * 60)
print("PHASE 1: Output projection layer FROZEN")
print("        Only the neural input layer can learn")
print("        (via gradients from Conversion bridge)")
print("=" * 60)

# Freeze projection layer
for p in proj_params:
    p.requires_grad_(False)

neural_optimizer = torch.optim.Adam(neural_params, lr=0.01)
# No projection optimizer needed — proj layer is frozen

frozen_losses = []
initial_test = evaluate(model, X_test, y_test)
frozen_losses.append(initial_test)
print(f"Initial Test MSE: {initial_test:.4f}")

for epoch in range(n_epochs):
    train_loss = train_epoch(model, X_train, y_train, neural_optimizer, proj_optimizer=None)
    test_loss = evaluate(model, X_test, y_test)
    frozen_losses.append(test_loss)
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:3d}/{n_epochs} | Train MSE: {train_loss:.4f} | Test MSE: {test_loss:.4f}")

# Check if neural layer weights actually changed
neural_w_changed = not torch.allclose(model.neural_layer.weight.data, init_neural_w)
proj_w_changed = not torch.allclose(model.proj_layer.weight.data, init_proj_w)

frozen_improvement = (frozen_losses[0] - frozen_losses[-1]) / frozen_losses[0] * 100

print(f"\n  Neural weights changed:     {neural_w_changed}")
print(f"  Projection weights changed: {proj_w_changed} (expected: False)")
print(f"  Test MSE improvement:       {frozen_improvement:.1f}%")

if frozen_improvement > 10 and neural_w_changed and not proj_w_changed:
    print("\n  ✓ PASS: Neural layer learned through Conversion bridge")
    print("          while output projection layer stayed frozen!")
elif not proj_w_changed and frozen_improvement <= 10:
    print("\n  ✗ FAIL: Neural layer did NOT learn through the bridge.")
else:
    print(f"\n  ? AMBIGUOUS: improvement={frozen_improvement:.1f}%, neural_changed={neural_w_changed}, proj_changed={proj_w_changed}")


# ══════════════════════════════════════════════════
# PHASE 2: Both layers train (sanity check)
# ══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 2: Both layers train (sanity check)")
print("=" * 60)

# Reset model
torch.manual_seed(42)
model2 = HybridNet(in_features=4, hidden_features=32, out_features=1)

neural_params2 = list(model2.neural_layer.parameters())
neural_ids2 = {id(p) for p in neural_params2}
proj_params2 = [p for p in model2.parameters() if id(p) not in neural_ids2]

neural_opt2 = torch.optim.Adam(neural_params2, lr=0.01)
proj_opt2 = ptorch_optim.AlternatingProjections(proj_params2)

both_losses = []
initial_test2 = evaluate(model2, X_test, y_test)
both_losses.append(initial_test2)
print(f"Initial Test MSE: {initial_test2:.4f}")

for epoch in range(n_epochs):
    train_loss = train_epoch(model2, X_train, y_train, neural_opt2, proj_opt2)
    test_loss = evaluate(model2, X_test, y_test)
    both_losses.append(test_loss)
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:3d}/{n_epochs} | Train MSE: {train_loss:.4f} | Test MSE: {test_loss:.4f}")

both_improvement = (both_losses[0] - both_losses[-1]) / both_losses[0] * 100
print(f"\n  Test MSE improvement: {both_improvement:.1f}%")


# ══════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Phase 1 (output frozen):  {frozen_losses[0]:.4f} → {frozen_losses[-1]:.4f}  ({frozen_improvement:+.1f}%)")
print(f"  Phase 2 (both train):     {both_losses[0]:.4f} → {both_losses[-1]:.4f}  ({both_improvement:+.1f}%)")
print()
if frozen_improvement > 10:
    print("  ✓ Conversion layer WORKS: the neural layer learned")
    print("    even with the output projection layer completely frozen.")
else:
    print("  ✗ Conversion layer BROKEN: no learning with frozen output layer.")
