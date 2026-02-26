from typing import Sequence
import torch
import torch.nn as nn
from ..core.ops import (
    MatMulProjection,
    SumReluProjection,
)

class Linear(nn.Module):
    """Linear (fully connected) layer without bias.
    Applies a linear transformation to input data.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # In pjax: Weight((in_features, out_features))
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, input):
        # Apply custom matmul projection
        return MatMulProjection.apply(input, self.weight)

class LinearBias(nn.Module):
    """Linear (fully connected) layer with bias.
    The bias is implemented by expanding the weight matrix.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features + 1, out_features))
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, input):
        # Append ones to input for bias computation
        ones = torch.ones((*input.shape[:-1], 1), dtype=input.dtype, device=input.device)
        augmented_input = torch.cat([input, ones], dim=-1)
        return MatMulProjection.apply(augmented_input, self.weight)

class ReLU(nn.Module):
    """Rectified Linear Unit with bias."""
    def __init__(self, features: int):
        super().__init__()

    def forward(self, *inputs):
        return SumReluProjection.apply(*inputs)

