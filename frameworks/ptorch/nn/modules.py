from typing import Sequence, Tuple, Union
import torch
import torch.nn as nn
from ..core.ops import (
    MatMulProjection,
    SumReluProjection,
    SimplexProjection,
    HardmaxProjection,
    Conversion as ConversionFn,
    MeanProjection,
    LayerNormProjection,
    DropoutProjection,
    SoftmaxProjection
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
        self.weight.is_projection = True
        self.proj_cache = {}
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, input, return_attention=False):
        # Apply custom matmul projection
        return MatMulProjection.apply(input, self.weight, self.proj_cache, self.alpha, self.g)

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
        self.weight.is_projection = True
        self.proj_cache = {}
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, input, return_attention=False):
        # Append ones to input for bias computation
        ones = torch.ones((*input.shape[:-1], 1), dtype=input.dtype, device=input.device)
        augmented_input = torch.cat([input, ones], dim=-1)
        return MatMulProjection.apply(augmented_input, self.weight, self.proj_cache, self.alpha, self.g)


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

class Mean(nn.Module):
    def __init__(self, dim):
        self.dim = dim
        super().__init__()
    
    def forward(self, input):
        return MeanProjection.apply(input, self.dim)

class LayerNorm(nn.Module):
    """Layer normalisation using LayerNormProjection.

    Forward: standard LayerNorm (no learnable affine parameters).
    Backward: projects inputs onto the locally linearised LayerNorm
    constraint graph instead of back-propagating real gradients.
    """
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, input):
        return LayerNormProjection.apply(input, self.eps)


class Dropout(nn.Module):
    """Projection-aware dropout.

    Forward: standard inverted dropout (zero with probability *p*,
    scale survivors by 1/(1-p)).
    Backward: projects inputs onto the dropout constraint graph.
    """
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, input):
        return DropoutProjection.apply(input, self.p, self.training)

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

    def __init__(self, model_features, qkv_features, heads, alpha=1.0, g=1.0, attention_type='simplex'):
        super().__init__()
        self.query_layer = Linear(model_features, heads * qkv_features, alpha=alpha, g=g)
        self.key_layer = Linear(model_features, heads * qkv_features, alpha=alpha, g=g)
        self.value_layer = Linear(model_features, heads * qkv_features, alpha=alpha, g=g)
        self.out_layer = Linear(heads * qkv_features, model_features, alpha=alpha, g=g)
        self.heads = heads
        self.attention_type = attention_type

    def _project_pairwise_matmul(self, left, right, omega=1.0):
        if left.ndim > 2 and right.ndim > 2:
            batch_shape = left.shape[:-2]
            if batch_shape != right.shape[:-2]:
                raise ValueError("Pairwise attention projection requires matching leading dimensions.")
                
        return MatMulProjection.apply(
            left,
            right,
            None, # no cache for pairwise
            self.query_layer.alpha,
            1,
            omega,
            10,   # num_steps = 10
            True  # pairwise
        )

    def forward(self, input, return_attention=False):
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
        scale = (q.shape[-1] ** 0.5)
        qk = self._project_pairwise_matmul(q, k.transpose(-2, -1), omega=scale)
        if self.attention_type == 'hardmax':
            qk = HardmaxProjection.apply(qk)
        elif self.attention_type == 'simplex':
            qk = SimplexProjection.apply(qk)
        else:
            qk = SoftmaxProjection.apply(qk)
        attention_weights = qk

        # Weighted sum of values
        o = self._project_pairwise_matmul(qk, v)

        # Merge heads: (batch_size, heads, seq_len, qkv) -> (batch_size, seq_len, heads*qkv)
        o = o.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        output = self.out_layer(o)
        if return_attention:
            return output, attention_weights
        return output
