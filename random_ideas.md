# Experimental Ideas in Ptorch

This document outlines the various random and experimental ideas, modules, and projections that have been implemented in the `ptorch` framework. 

## 1. Experimental Activations

### Gapped Step (`GappedStep`)
A Step activation function that features a "dead zone" or gap.
- **Concept:** Outputs `+1` for $x \ge \delta/2$ and `-1` for $x \le -\delta/2$, but returns `0` in the gap region in the forward pass.
- **Projection Behavior:** The backward projection operator enforces a gap constraint ($|y| \ge \delta/2$) in the output space, forcing the network to commit to decisive output values by snapping those within the gap to the nearest boundary.

### Pure Gap (`Gap`)
- **Concept:** The forward pass is merely a pass-through identity.
- **Projection Behavior:** The projection backward enforces that outputs must lie outside a gap region ($|y| \ge \delta/2$). This forces the network to commit to a definitive decision rather than hovering near zero.

### Infinite-Ladder Quantization (`Quantize`)
- **Concept:** An infinite-ladder staircase function where inputs are rounded to the nearest multiple of a given step size: $f(x) = \text{step} \times \text{round}(x / \text{step})$. 
- **Projection Behavior:** Unlike bounded quantizers that saturate, this has infinitely many rungs. The backward pass computes the exact Euclidean projection onto the graph of the staircase, providing a geometric training signal.

### Quantized ReLU (`QuantizeReLU`)
- **Concept:** A ReLU function combined with quantization steps.
- **Projection Behavior:** Backpropagation correctly handles the non-negative constraints on quantization rungs.

### Squared ReLU (`ReLUSquared`)
- **Concept:** Uses $f(x) = \max(0, x)^2$.
- **Projection Behavior:** Backward pass performs an exact Euclidean projection onto the Squared ReLU constraint graph.

### Sum ReLU (`SumReLU`)
- **Concept:** Aggregates and applies ReLU over multiple inputs.

## 2. Experimental Linear & Dense Layers

### Orthogonal Linear Layer (`LinearOrth`)
- **Concept:** A linear layer that maintains a square weight matrix, initialized to be orthogonal.
- **Projection Behavior:** Uses a closed-form Orthogonal Procrustes projection (`OrthogonalRotationProjection`) to ensure the weight matrix updates remain an orthogonal rotation.

### Hybrid Linear Layer (`LinearHybrid`)
- **Concept:** A drop-in replacement for standard Linear layers using a hybrid gradient-projection approach.
- **Projection Behavior:** Upstream gradient flows using the standard chain-rule gradient ($\partial L/\partial A = Z_{\text{grad}} B^T$), avoiding the vanishing-target problem. The weights receive a standard gradient ($A^T Z_{\text{grad}}$) in `p.grad` and can be fed to `ProjectionMuon`, where Muon's Newton-Schulz step auto-scales and orthogonalizes the gradient before updating.

### Absolute Linear Layer (`LinearAbs`)
- **Concept:** A fused Linear and Absolute value layer.
- **Projection Behavior:** Uses exact $\ell_\infty$ projection (`AbsBilinearProjectionLinf`) for the combined operation.

## 3. Projection-Aware Normalization & Attention

### Affine Batch Normalization (`BatchNorm`)
- **Concept:** Batch normalization that projects the input tensor *as well as* the learnable scale and shift parameters using full projection logic.

### Causal Self Attention
- **Concept:** A projection-aware implementation of causal self-attention.
- **Features:** Supports RoPE (Rotary Positional Embeddings), RMSNorm, Grouped Query Attention (GQA), and Causal Masking, all utilizing projection logic internally.

### Multi-Head Attention
- **Concept:** MHA extended with GQA support.
- **Projection Behavior:** Uses `SimplexProjection` instead of standard softmax for attention weights, and `MatMulExactProjection` for projected matrix multiplications.

## 4. Alternate Target Projections

- **`Simplex` / `SimplexProjection`**: Projects targets onto a simplex, often used as an alternative to softmax for attention distributions.
- **`Hardmax` / `HardmaxProjection`**: Hardmax function projection.
- **`DropoutProjection`**: Projects inputs onto the dropout constraint graph (zeroing or scaling up based on the dropout mask constraint).
