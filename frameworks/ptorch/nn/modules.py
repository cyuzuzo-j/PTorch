from typing import Sequence, Tuple, Union
import torch
import torch.nn as nn
from ..core.ops import (
    MatMulProjection,
    MatMulExactProjection,
    SumReluProjection,
    SimplexProjection,
    Conversion as ConversionFn
)

class Linear(nn.Module):
    """Linear (fully connected) layer without bias.
    Applies a linear transformation to input data.
    """
    def __init__(self, in_features: int, out_features: int, alpha: float = 1.0, g: float = 1.0, num_iters: int = 5):
        super().__init__()
        self.alpha = alpha
        self.g = g
        self.num_iters = num_iters
        # In pjax: Weight((in_features, out_features))
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, input):
        # Apply custom matmul projection
        return MatMulExactProjection.apply(input, self.weight, self.alpha, self.g)

class LinearBias(nn.Module):
    """Linear (fully connected) layer with bias.
    The bias is implemented by expanding the weight matrix.
    """
    def __init__(self, in_features: int, out_features: int, alpha: float = 1.0, g: float = 1.0, num_iters: int = 1):
        super().__init__()
        self.alpha = alpha
        self.g = g
        self.num_iters = num_iters
        self.weight = nn.Parameter(torch.empty(in_features + 1, out_features))
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, input):
        # Append ones to input for bias computation
        ones = torch.ones((*input.shape[:-1], 1), dtype=input.dtype, device=input.device)
        augmented_input = torch.cat([input, ones], dim=-1)
        return MatMulExactProjection.apply(augmented_input, self.weight, self.alpha, self.g)


class LinearExact(nn.Module):
    """Linear layer using exact independent bilinear projection.

    Each dot product a_i · b_j is projected independently; shared variables
    are reconciled by consensus averaging over all (i, j) pairs. Equivalent
    to ``pjax.matmul_exact`` / ``pjax.matmul_slower``: better convergence at
    the cost of O(M×N×K) memory per backward pass.

    Args:
        in_features: number of input features.
        out_features: number of output features.
        alpha: stiffness for weight update (default 1.0).
        g: stiffness for output target (default 1.0).
    """
    def __init__(self, in_features: int, out_features: int, alpha: float = 1.0, g: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.g = g
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, input):
        return MatMulExactProjection.apply(input, self.weight, self.alpha, self.g)


class Conv2D(nn.Module):
    """2D convolutional layer via patch extraction and linear projection.

    Mirrors the pjax Conv2D architecture:
      1. Extract patches using ConvPatchProjection → (N, H_out, W_out, C_in*kH*kW)
      2. Apply LinearBias to project patches     → (N, H_out, W_out, out_channels)

    Input:  (N, C_in, H, W)   — standard PyTorch NCHW
    Output: (N, out_channels, H_out, W_out)

    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_size: size of the convolution kernel (int or (h, w)).
        stride: stride of the convolution (int or (h, w)).
        padding: padding added to input (int, (h, w), or 'same').
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int], str] = 0,
        alpha: float = 1.0,
        g: float = 1.0,
        num_iters: int = 1,
    ):
        super().__init__()
        # Normalize to tuples
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_mode = padding

        # LinearBias projects from patch features to output channels
        # Matches pjax: LinearBias(in_channels * kH * kW, out_channels)
        kH, kW = kernel_size
        self.linear = LinearBias(in_channels * kH * kW, out_channels, alpha=alpha, g=g, num_iters=num_iters)

    def _resolve_padding(self, H: int, W: int) -> Tuple[int, int]:
        """Resolve padding to explicit (pad_h, pad_w) values."""
        if isinstance(self.padding_mode, str):
            if self.padding_mode.lower() == 'same':
                kH, kW = self.kernel_size
                sH, sW = self.stride
                pad_h = max(0, (H - 1) * sH + kH - H) // 2
                pad_w = max(0, (W - 1) * sW + kW - W) // 2
                return (pad_h, pad_w)
            elif self.padding_mode.lower() == 'valid':
                return (0, 0)
            else:
                raise ValueError(f"Unknown padding mode: {self.padding_mode}")
        elif isinstance(self.padding_mode, int):
            return (self.padding_mode, self.padding_mode)
        else:
            return tuple(self.padding_mode)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        N, C, H, W = input.shape
        padding = self._resolve_padding(H, W)

        # Step 1: patch extraction → (N, H_out, W_out, C*kH*kW)
        patches = ConvPatchProjection.apply(input, self.kernel_size, self.stride, padding)

        # Step 2: linear projection → (N, H_out, W_out, out_channels)
        out = self.linear(patches)

        # Step 3: permute to NCHW → (N, out_channels, H_out, W_out)
        out = out.permute(0, 3, 1, 2)
        return out


class ReLU(nn.Module):
    """Rectified Linear Unit with bias."""
    def __init__(self, features: int):
        super().__init__()

    def forward(self, *inputs):
        return SumReluProjection.apply(*inputs)

class Step(nn.Module):
    """Step activation function with bias."""
    def __init__(self, features: int):
        super().__init__()
        
    def forward(self, *inputs):
        from ..core.ops import StepProjection # Avoid circular import if needed or just use it here
        return StepProjection.apply(*inputs)

class Step_NB(nn.Module):
    """Step activation function without bias."""
    def __init__(self):
        super().__init__()
        
    def forward(self, *inputs):
        from ..core.ops import StepProjection
        return StepProjection.apply(*inputs)

class Simplex(nn.Module):
    """Simplex activation function."""
    def __init__(self, features: int = 0):
        super().__init__()
        
    def forward(self, input):
        return SimplexProjection.apply(input)

class Conversion(nn.Module):
    """Bridge layer: converts projection targets into real gradients.
    
    Forward: identity (pass-through).
    Backward: receives projection target, returns gradient (input - target)
    so that upstream gradient-based layers (e.g. Embedding) can learn.
    """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return ConversionFn.apply(input)

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for transformer architectures.

    Implements the multi-head attention mechanism as described in "Attention
    Is All You Need" (Vaswani et al., 2017). Splits the input into multiple
    attention heads and applies scaled dot-product attention.

    Uses SimplexProjection instead of softmax for attention weights,
    and MatMulExactProjection for projected matrix multiplications.

    Args:
        model_features: dimensionality of input and output features.
        qkv_features: dimensionality of query, key, and value vectors per head.
        heads: number of attention heads.

    Attributes:
        query_layer: linear layer for computing queries.
        key_layer: linear layer for computing keys.
        value_layer: linear layer for computing values.
        out_layer: output projection layer.
        heads: number of attention heads.
    """

    def __init__(self, model_features, qkv_features, heads, alpha=1.0, g=1.0):
        super().__init__()
        self.query_layer = Linear(model_features, heads * qkv_features, alpha=alpha, g=g)
        self.key_layer = Linear(model_features, heads * qkv_features, alpha=alpha, g=g)
        self.value_layer = Linear(model_features, heads * qkv_features, alpha=alpha, g=g)
        self.out_layer = Linear(heads * qkv_features, model_features, alpha=alpha, g=g)
        self.heads = heads

    def forward(self, input):
        """Compute multi-head attention.

        Args:
            input: tensor of shape ``(batch_size, seq_len, model_features)``.

        Returns:
            tensor of shape ``(batch_size, seq_len, model_features)``.
        """
        q = self.query_layer(input)
        k = self.key_layer(input)
        v = self.value_layer(input)

        # Split into heads: (batch_size, seq_len, heads*qkv) -> (batch_size, heads, seq_len, qkv)
        batch_size, seq_len = input.shape[:2]

        def split_heads(x):
            return x.reshape(batch_size, seq_len, self.heads, -1).permute(0, 2, 1, 3)

        q, k, v = map(split_heads, (q, k, v))

        # Compute attention scores
        scale = 1.0 / (q.shape[-1] ** 0.5)
        qk = MatMulExactProjection.apply(q, k.transpose(-2, -1), self.query_layer.alpha, self.query_layer.g)
        qk = qk + scale  # simple scaling via addition (matches pjax)
        qk = SimplexProjection.apply(qk)

        # Weighted sum of values
        o = MatMulExactProjection.apply(qk, v, self.query_layer.alpha, self.query_layer.g)

        # Merge heads: (batch_size, heads, seq_len, qkv) -> (batch_size, seq_len, heads*qkv)
        o = o.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        return self.out_layer(o)
