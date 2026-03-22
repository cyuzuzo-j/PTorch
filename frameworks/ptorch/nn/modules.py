from typing import Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.ops import (
    MatMulProjection,
    MatMulProjectionLinf,
    ReLULInfinityProjection,
    SumReLUProjection,
    StepProjection,
    ReLUProjection,
    LeakyReLUProjection,
    simplex_op_pt,
    SimplexProjection,
    HardmaxProjection,
    Conversion as ConversionFn,
    MeanProjection,
    LayerNormProjection,
    DropoutProjection,
    SoftmaxProjection,
    AddProjection,
    AffineBatchNormProjection,
)
from .. import config

class Linear(nn.Linear):
    """Linear (fully connected) layer with optional projection support.
    
    Applies a linear transformation to input data. Supports both L2 (Frobenius)
    and L∞ (Chebyshev) projections for backward pass through the constraint.

    Args:
        in_features:  number of input features.
        out_features: number of output features.
        bias:         if True (default), adds a learnable bias.
        alpha:        L2 projection parameter (default 1.0).
        g:            Z-slack weight in the L∞ projection (default 1.0).
        omega:        output scale divisor (default 1.0).
        num_iters:    Newton iterations inside the projection (default 5).
        residual:     if True, forward computes ``(x - x @ W) / omega``
                      and the residual isomorphism is applied in the projection.
        norm:         projection norm type, 'l2' for Frobenius or 'linf' for Chebyshev (default 'l2').
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        alpha: float = 1.0,
        g: float = 1.0,
        omega: float = 1.0,
        num_iters: int = 5,
        residual: bool = False,
        norm: str = 'l2',
    ):
        # Never let nn.Linear create the bias; we create our own 2D version below.
        super().__init__(in_features, out_features, bias=False)
        self.alpha = alpha
        self.g = g
        self.omega = omega
        self.num_iters = num_iters
        self.use_bias = bias
        self.proj_cache: dict = {}
        self.residual = residual
        self.norm = norm.lower()
        
        if self.norm not in ('l2', 'linf'):
            raise ValueError(f"norm must be 'l2' or 'linf', got '{norm}'")
        
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='linear')
        if self.residual:
            with torch.no_grad():
                identity = torch.eye(
                    out_features,
                    in_features,
                    device=self.weight.device,
                    dtype=self.weight.dtype,
                )
                self.weight.copy_(identity - self.weight)
        
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(1, out_features))

    def forward(self, input, norm=None):
        norm = norm or self.norm
        if config.use_projections:
            weight_t = self.weight.T
            if self.use_bias:
                input = F.pad(input, (0, 1), value=1.0)
                weight_t = torch.cat([weight_t, self.bias], dim=0)
            
            if self.residual:
                in_dim = input.shape[-1]
                out_dim = weight_t.shape[-1]
                padded_dim = max(in_dim, out_dim)

                if padded_dim > in_dim:
                    input = F.pad(input, (0, padded_dim - in_dim), value=0.0)

                if padded_dim != in_dim or padded_dim != out_dim:
                    padded_weight_t = weight_t.new_zeros((padded_dim, padded_dim))
                    padded_weight_t[:in_dim, :out_dim] = weight_t
                    weight_t = padded_weight_t

                if self.norm == 'linf':
                    output = MatMulProjectionLinf.apply(
                        input, weight_t,
                        self.num_iters, self.g, self.omega,
                        self.proj_cache, False, True,
                    )
                else:  # l2
                    output = MatMulProjection.apply(
                        input, weight_t,
                        self.num_iters, self.alpha, self.g, self.omega,
                        self.proj_cache, False, True,
                    )
                return output[..., :self.out_features]
            
            if norm == 'linf':
                return MatMulProjectionLinf.apply(
                    input, weight_t,
                    self.num_iters, self.g, self.omega,
                    self.proj_cache, False, self.residual,
                )
            else:  # l2
                return MatMulProjection.apply(
                    input, weight_t,
                    self.num_iters, self.alpha, self.g, self.omega,
                    self.proj_cache, False, self.residual,
                )

        output = F.linear(input, self.weight)
        if self.use_bias:
            output = output + self.bias
        return output / self.omega


class ReLU(nn.ReLU):
    def __init__(self, inplace: bool = False, norm="l2"):
        super().__init__(inplace=inplace)
        self.norm = norm

    def forward(self, input, norm=None):
        norm = norm or self.norm
        if config.use_projections:
            if norm == 'linf':
                return ReLULInfinityProjection.apply(input)
            return ReLUProjection.apply(input)
        if norm == 'linf':
            raise ValueError("L∞ projection for ReLU is not supported in gradient mode")
        return super().forward(input)


class LeakyReLU(nn.LeakyReLU):
    def __init__(self, negative_slope: float = 0.01, inplace: bool = False):
        super().__init__(negative_slope=negative_slope, inplace=inplace)

    def forward(self, input):
        if config.use_projections:
            return LeakyReLUProjection.apply(input, self.negative_slope)
        return super().forward(input)

class SumReLU(nn.Module):
    """Rectified Linear Unit."""
    def __init__(self):
        super().__init__()

    def forward(self, *inputs):
        if config.use_projections:
            return SumReLUProjection.apply(*inputs)
        return torch.relu(sum(inputs))

class Step(nn.Module):
    """Step activation function."""
    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        if config.use_projections:
            return StepProjection.apply(input)
        return torch.where(input >= 0, torch.tensor(1.0, dtype=input.dtype, device=input.device), 
                           torch.tensor(-1.0, dtype=input.dtype, device=input.device))


class Simplex(nn.Module):
    """Simplex activation function."""
    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        if config.use_projections:
            return SimplexProjection.apply(input)
        Warning.warn("Simplex is not supported in gradient mode")
        return simplex_op_pt(input)

class Conversion(nn.Module):
    """Bridge layer: converts projection targets into real gradients.
    
    Forward: identity (pass-through).
    Backward: receives projection target, returns gradient (input - target)
    so that upstream gradient-based layers (e.g. Embedding) can learn.
    """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        if config.use_projections:
            return ConversionFn.apply(input)
        return input


class Mean(nn.Module):
    def __init__(self, dim):
        self.dim = dim
        super().__init__()
    
    def forward(self, input):
        if config.use_projections:
            return MeanProjection.apply(input, self.dim)
        return torch.mean(input, dim=self.dim)

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
        if config.use_projections:
            return LayerNormProjection.apply(input, self.eps)
        mu = input.mean(dim=-1, keepdim=True)
        var = input.var(dim=-1, keepdim=True, unbiased=False)
        sigma = torch.sqrt(var + self.eps)
        return (input - mu) / sigma


class Add(nn.Module):
    """Residual addition using AddProjection.
    
    Forward: returns x1 + x2.
    Backward: projects x1 and x2 onto the addition constraint graph.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        if config.use_projections:
            return AddProjection.apply(x1, x2)
        return x1 + x2

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
        if config.use_projections:
            return DropoutProjection.apply(input, self.p, self.training)
        if self.training and self.p > 0.0:
            mask = (torch.rand_like(input) > self.p).to(input.dtype)
            return input * mask * (1.0 / (1.0 - self.p))
        return input


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

    def __init__(self, model_features, qkv_features, heads, alpha=1.0, g=1.0, attention_type='simplex', norm="l2"):
        super().__init__()
        self.query_layer = Linear(model_features, heads * qkv_features, alpha=alpha, g=g, norm=norm)
        self.key_layer = Linear(model_features, heads * qkv_features, alpha=alpha, g=g, norm=norm)
        self.value_layer = Linear(model_features, heads * qkv_features, alpha=alpha, g=g, norm=norm)
        self.out_layer = Linear(heads * qkv_features, model_features, alpha=alpha, g=g, norm=norm)
        self.heads = heads
        self.attention_type = attention_type

    def _project_pairwise_matmul(self, left, right, omega=1.0):
        if not config.use_projections:
            return (left @ right) / omega
            
        if left.ndim > 2 and right.ndim > 2:
            batch_shape = left.shape[:-2]
            if batch_shape != right.shape[:-2]:
                raise ValueError("Pairwise attention projection requires matching leading dimensions.")
                
        return MatMulProjection.apply(
            left,
            right,
            5,
            self.query_layer.alpha,
            1,
            omega,
            None,
            True,
            False,
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
            if config.use_projections:
                qk = HardmaxProjection.apply(qk)
            else:
                from ..core.ops import hardmax_op_pt
                qk = hardmax_op_pt(qk)
        elif self.attention_type == 'simplex':
            if config.use_projections:
                qk = SimplexProjection.apply(qk)
            else:
                from ..core.ops import simplex_op_pt
                qk = simplex_op_pt(qk)
        else:
            if config.use_projections:
                qk = SoftmaxProjection.apply(qk)
            else:
                qk = F.softmax(qk, dim=-1)
        attention_weights = qk

        # Weighted sum of values
        o = self._project_pairwise_matmul(qk, v)

        # Merge heads: (batch_size, heads, seq_len, qkv) -> (batch_size, seq_len, heads*qkv)
        o = o.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        output = self.out_layer(o)
        if return_attention:
            return output, attention_weights
        return output


class BatchNorm(nn.Module):
    """
    Affine Batch normalisation using full projection logic.
    Projects the input tensor AND the learnable scale/shift parameters.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, num_steps: int = 3):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.num_steps = num_steps
        
        # The scale (gamma) and shift (beta)
        self.weight = nn.Parameter(torch.ones(1,num_features))
        self.bias = nn.Parameter(torch.zeros(1,num_features))

    def forward(self, input):
        if config.use_projections:
            return AffineBatchNormProjection.apply(
                input, self.weight, self.bias, self.eps, self.num_steps
            )
            
        # Standard execution if projections are turned off
        mu = input.mean(dim=0, keepdim=True)
        var = input.var(dim=0, keepdim=True, unbiased=False)
        x_hat = (input - mu) / torch.sqrt(var + self.eps)
        return self.weight * x_hat + self.bias