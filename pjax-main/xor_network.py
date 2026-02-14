import jax
import jax.numpy as jnp
import pjax
from pjax import nn, optim
from pjax.optim import get_graph, plot_graph
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

print("="*60)
print("XOR Network Training with PJAX")
print("="*60)

# Define XOR neural network with 1 hidden layer (2 neurons)
class XORNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 2)  # Input: 2, Hidden: 2 neurons
        self.relu1 = nn.ReLU(2)
        self.output = nn.Linear(2, 1)  # Output: 1
        self.relu2 = nn.ReLU(1)  # ReLU output activation
    
    def __call__(self, x):
        x = self.hidden(x)
        x = self.relu1(x)
        x = self.output(x)
        x = self.relu2(x)
        return x

# XOR dataset
X = jnp.array([[0.0, 0.0],
               [0.0, 1.0],
               [1.0, 0.0],
               [1.0, 1.0]])

y = jnp.array([[0.0],
               [1.0],
               [1.0],
               [0.0]])

print("\nInitializing model...")
key = jax.random.key(42)
model = XORNetwork()
params = model.init(key)

# Initialize optimizer (using DouglasRachford from PJAX)
optimizer = optim.DouglasRachford(steps_per_update=1)

print("Training XOR Network (100 epochs)...")

# Training loop WITHOUT JIT for debugging
def train_step(params, x, y_true):
    def apply_fn(params):
        y_pred = model.apply(params, x)
        return pjax.means_squared_error(y_pred, y_true)
    
    updated_params, loss = optimizer.update(apply_fn, params)
    return updated_params, loss

num_epochs = 1
loss_history = []

for epoch in range(num_epochs):
    params, loss = train_step(params, X, y)
    loss_val = float(loss)
    loss_history.append(loss_val)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}, Loss: {loss_val:.6f}")

print(f"\nFinal Loss: {loss_history[-1]:.6f}")

# Test predictions
predictions = model.apply(params, X)
print("\nPredictions:")
for i in range(len(X)):
    print(f"Input: [{X[i,0]:.1f}, {X[i,1]:.1f}] → Target: {y[i,0]:.1f}, Prediction: {predictions[i,0]:.4f}")

# ===== PJAX COMPUTATIONAL GRAPH VISUALIZATION =====
print("\n" + "="*60)
print("GENERATING COMPUTATIONAL GRAPH")
print("="*60)

def forward_computation(params):
    y_pred = model.apply(params, X)
    return pjax.means_squared_error(y_pred, y)

# Convert params to Parameter objects
params_for_graph = {name: pjax.core.computation.Parameter(value, name=name) 
                    for name, value in params.items()}
computation = forward_computation(params_for_graph)

# Get the computational graph
graph = get_graph(computation)

print(f"\nGraph Statistics:")
print(f"  Nodes: {graph.number_of_nodes()}")
print(f"  Edges: {graph.number_of_edges()}")

# Count node types
node_types = {}
for node in graph.nodes:
    node_type = type(node).__name__
    node_types[node_type] = node_types.get(node_type, 0) + 1

print(f"\nNode Types:")
for node_type, count in sorted(node_types.items()):
    print(f"  {node_type}: {count}")

# Save computational graph using PJAX's built-in plot_graph()
print("\nSaving PJAX computational graph (built-in)...")
plt.figure(figsize=(14, 12))
plot_graph(graph)
plt.title('PJAX Computational Graph - XOR Network', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('/home/cyuzuzo/thesisV5/pjax-main/xor_pjax_builtin_graph.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: xor_pjax_builtin_graph.png")

# Convert NetworkX graph to TikZ using built-in function
print("\nConverting computational graph to TikZ...")
try:
    from networkx.drawing.nx_latex import to_latex
    tikz_code = to_latex(graph, pos=pos if 'pos' in locals() else None, 
                         tikz_options="scale=1.5")
    
    # Save TikZ code to file
    with open('/home/cyuzuzo/thesisV5/pjax-main/xor_pjax_graph.tex', 'w') as f:
        f.write(tikz_code)
    print("✓ Saved: xor_pjax_graph.tex")
except ImportError:
    print("⚠ NetworkX LaTeX export not available, skipping TikZ conversion")
except Exception as e:
    print(f"⚠ TikZ conversion failed: {e}")

# Create comprehensive visualization
print("\nCreating comprehensive visualization...")
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Custom PJAX Computational Graph with labels
ax1 = fig.add_subplot(gs[0:2, 0:2])

try:
    A, B = nx.bipartite.sets(graph)
    node_color = ["#2E86AB" if node in A else "#C73E1D" for node in graph.nodes]
except:
    node_color = ["#2E86AB" for _ in graph.nodes]

labels = {}
for node in graph.nodes:
    node_type = type(node).__name__
    if hasattr(node, 'name'):
        labels[node] = f"{node.name}"
    elif hasattr(node, 'operation'):
        op_name = node.operation.__name__ if hasattr(node.operation, '__name__') else str(node.operation)
        labels[node] = f"{op_name}"
    else:
        labels[node] = node_type[:8]

pos = nx.spring_layout(graph, iterations=2000, seed=42)
nx.draw(graph, pos, labels=labels, with_labels=True, node_color=node_color, 
        node_size=1000, font_size=8, font_weight='bold', font_color='white',
        edge_color='gray', arrows=True, ax=ax1, arrowsize=20, width=2)
ax1.set_title('PJAX Computational Graph', fontsize=14, fontweight='bold', pad=10)

# 2. Loss curve
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(loss_history, linewidth=2, color='#2E86AB')
ax2.set_xlabel('Epoch', fontsize=10)
ax2.set_ylabel('Loss (MSE)', fontsize=10)
ax2.set_title('Training Loss', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

# 3. Decision boundary
ax3 = fig.add_subplot(gs[1, 2])
x_min, x_max = -0.5, 1.5
y_min, y_max = -0.5, 1.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
grid_points = jnp.array(np.c_[xx.ravel(), yy.ravel()])
Z = model.apply(params, grid_points)
Z = np.array(Z).reshape(xx.shape)

contour = ax3.contourf(xx, yy, Z, levels=15, cmap='RdYlBu_r', alpha=0.8)
plt.colorbar(contour, ax=ax3, label='Output', fraction=0.046)
ax3.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='RdYlBu_r', 
           s=150, edgecolors='black', linewidth=2, marker='o')
ax3.set_xlabel('$x_1$', fontsize=10)
ax3.set_ylabel('$x_2$', fontsize=10)
ax3.set_title('Decision Boundary', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. Neural Network Architecture
ax4 = fig.add_subplot(gs[2, :])
ax4.axis('off')
ax4.set_xlim(0, 4)
ax4.set_ylim(0, 3.5)

# Input layer
input_y = [1, 2.5]
for i, y_pos in enumerate(input_y):
    circle = plt.Circle((0.5, y_pos), 0.25, color='#A23B72', ec='black', linewidth=2)
    ax4.add_patch(circle)
    ax4.text(0.5, y_pos, f'$x_{i+1}$', ha='center', va='center', fontsize=11, fontweight='bold', color='white')

# Hidden layer
hidden_y = [1, 2.5]
for i, y_pos in enumerate(hidden_y):
    circle = plt.Circle((2, y_pos), 0.25, color='#F18F01', ec='black', linewidth=2)
    ax4.add_patch(circle)
    ax4.text(2, y_pos, f'$h_{i+1}$', ha='center', va='center', fontsize=11, fontweight='bold', color='white')

# Output layer
output_y = 1.75
circle = plt.Circle((3.5, output_y), 0.25, color='#C73E1D', ec='black', linewidth=2)
ax4.add_patch(circle)
ax4.text(3.5, output_y, '$y$', ha='center', va='center', fontsize=11, fontweight='bold', color='white')

# Connections input to hidden (using placeholder weights)
for i, in_y in enumerate(input_y):
    for j, hid_y in enumerate(hidden_y):
        weight = 1.0  # Placeholder weight
        linewidth = min(abs(weight) * 1.5, 4)
        color = '#2E86AB' if weight > 0 else '#C73E1D'
        ax4.plot([0.75, 1.75], [in_y, hid_y], color=color, linewidth=linewidth, alpha=0.7)
        mid_x, mid_y = 1.25, (in_y + hid_y) / 2
        ax4.text(mid_x, mid_y, f'{weight:.2f}', fontsize=7, ha='center', 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='gray'))

# Connections hidden to output (using placeholder weights)
for j, hid_y in enumerate(hidden_y):
    weight = 1.0  # Placeholder weight
    linewidth = min(abs(weight) * 1.5, 4)
    color = '#2E86AB' if weight > 0 else '#C73E1D'
    ax4.plot([2.25, 3.25], [hid_y, output_y], color=color, linewidth=linewidth, alpha=0.7)
    mid_x, mid_y = 2.75, (hid_y + output_y) / 2
    ax4.text(mid_x, mid_y, f'{weight:.2f}', fontsize=7, ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='gray'))

# Layer labels
ax4.text(0.5, 3.2, 'Input', ha='center', fontsize=10, fontweight='bold')
ax4.text(2, 3.2, 'Hidden (ReLU)', ha='center', fontsize=10, fontweight='bold')
ax4.text(3.5, 3.2, 'Output (ReLU)', ha='center', fontsize=10, fontweight='bold')
ax4.set_title('Neural Network Architecture (1 Hidden Layer, 2 Neurons)', fontsize=12, fontweight='bold')

plt.savefig('/home/cyuzuzo/thesisV5/pjax-main/xor_network_pjax.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: xor_network_pjax.png")

# Print TikZ-friendly format
print("\n" + "="*60)
print("TIKZ CONVERSION DATA")
print("="*60)
print("\nNetwork Structure:")
print("  - Input neurons: 2 (x1, x2)")
print("  - Hidden neurons: 2 (h1, h2) with ReLU activation")
print("  - Output neurons: 1 (y) with ReLU activation")

print("\nWeights (Input → Hidden): [Placeholder values]")
for i in range(2):
    for j in range(2):
        print(f"  w{{x{i+1}→h{j+1}}} = 1.0000")

print("\nBiases (Hidden Layer): [Placeholder values]")
for j in range(2):
    print(f"  b{{h{j+1}}} = 0.0000")

print("\nWeights (Hidden → Output): [Placeholder values]")
for j in range(2):
    print(f"  w{{h{j+1}→y}} = 1.0000")

print("\nBias (Output Layer): [Placeholder value]")
print(f"  b{{y}} = 0.0000")

print("\nTikZ Node Coordinates:")
print("  Input layer (x=0.5):")
for i, y_pos in enumerate(input_y):
    print(f"    x{i+1}: (0.5, {y_pos})")
print("  Hidden layer (x=2):")
for i, y_pos in enumerate(hidden_y):
    print(f"    h{i+1}: (2, {y_pos})")
print(f"  Output layer: y: (3.5, {output_y})")

print("\nTikZ Styling:")
print("  Positive weights: Blue (#2E86AB)")
print("  Negative weights: Red (#C73E1D)")
print("  Line width: |weight| × 1.5 (max 4)")

print("\n" + "="*60)
print("✓ ALL VISUALIZATIONS COMPLETE!")
print("="*60)
print("\nGenerated files:")
print("  1. xor_pjax_builtin_graph.png - PJAX built-in graph visualization")
print("  2. xor_network_pjax.png - Comprehensive visualization")
print("  3. xor_pjax_graph.tex - TikZ code for computational graph")
print("="*60)
