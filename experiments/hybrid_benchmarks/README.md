# Hybrid Benchmark: Ptorch MLP with PyTorch CNN Feature Extractor

This experiment intends to run a hybrid architecture where the first few layers are standard PyTorch convolutional layers, and the final linear features head are custom `ptorch` projection layers. This helps evaluate the modularity of the projection framework and its capacity to connect standard gradients onto custom constraints via `GradientToProjection`. The `GradientToProjection` module translates the real backpropagated PyTorch gradient into a target state for projection optimization.

### Architectures
- **Standard Baseline** (`bench_torch.py`): Conv2d -> MaxPool2d -> Conv2d -> MaxPool2d -> Flatten -> Linear -> ReLU -> Linear
- **Hybrid System** (`bench_ptorch.py`): Conv2d -> MaxPool2d -> Conv2d -> MaxPool2d -> Flatten -> *GradientToProjection* -> LinearBias (Ptorch) -> Simplex (Ptorch) -> LinearBias (Ptorch)
