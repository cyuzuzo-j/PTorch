from typing import Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.ops import * 
from .. import config

class ProjectionModule(nn.Module):
    """Base class for projection-aware modules."""
    def __init__(self, outputs):
        super().__init__()
        self.outputs = outputs
        self.projection_forward_cache = [None for _ in range(outputs)]
    
class Linear(ProjectionModule):
    """Projection-only linear layer following the old main-style matmul path.

    Supports optional affine bias via input augmentation and optional residual
    projection mode. Accepts L2, Linf, L1, and general Lp norms.
    Fixed to include mathematically correct residual padding and initialization.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        alpha: float = 1.0,
        g: float = 1.0,
        omega: float = 1.0,
        num_iters: int = 1,
        residual: bool = False,
        dtp: bool = False,
        norm = 'l2',
    ):
        super().__init__(outputs=1)
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.g = g
        self.omega = omega
        self.num_iters = int(num_iters)
        self.use_bias = bias
        self.residual = residual
        self.dtp = dtp
        self.norm = norm
        self.proj_cache: dict = {}

        # Separate parameters to ensure optimizers (e.g., AdamW) can exclude bias from weight decay
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(1, out_features))
        else:
            self.register_parameter('bias', None)

        # 1. Correct Initialization Variance:
        # Now matches nn.Linear shape (out_features, in_features) natively
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='linear')
        
        # 2. Residual Identity mapping
        if self.residual:
            with torch.no_grad():
                identity = torch.eye(
                    out_features, in_features, 
                    device=self.weight.device, 
                    dtype=self.weight.dtype
                )
                self.weight.copy_(identity - self.weight)

    def forward(self, input, norm=None):
        norm_type = norm if norm is not None else self.norm

        if not config.use_projections: # Assumes config is in scope
            # Standard gradient path
            b = self.bias.squeeze(0) if self.use_bias else None
            output = F.linear(input, self.weight, b)
            return output / self.omega

        # Speed/Memory Efficiency: Use F.pad instead of torch.cat for input bias augmentation
        projected_input = input
        weight_matrix = self.weight.T

        if self.use_bias:
            projected_input = F.pad(input, (0, 1), value=1.0)
            weight_matrix = torch.cat([weight_matrix, self.bias], dim=0)

        # Mathematically Correct Residual Padding
        if self.residual:
            in_dim = projected_input.shape[-1]
            out_dim = weight_matrix.shape[-1]
            padded_dim = max(in_dim, out_dim)

            if padded_dim > in_dim:
                projected_input = F.pad(projected_input, (0, padded_dim - in_dim), value=0.0)

            if padded_dim != in_dim or padded_dim != out_dim:
                padded_weight = weight_matrix.new_zeros((padded_dim, padded_dim))
                padded_weight[:in_dim, :out_dim] = weight_matrix
                weight_matrix = padded_weight

        # Dispatch to the correct projection function
        if norm_type == 'linf':
            output = MatMulProjectionLinf.apply(
                projected_input, weight_matrix, self.num_iters,
                self.g, self.omega, # Aligned with LinearLinf (ignores config.projection_g)
                self.proj_cache, False, self.residual,
            )
        elif self.dtp:
            output = MatMulProjectionDTP.apply(
                projected_input, weight_matrix, self.num_iters,
                self.alpha * config.projection_alpha,
                self.g * config.projection_g, self.omega,
                self.proj_cache, False, self.residual,
            )                
        else:
            output = MatMulProjection.apply(
                projected_input, weight_matrix, self.num_iters,
                self.alpha * config.projection_alpha,
                self.g * config.projection_g, self.omega,
                self.proj_cache, False, self.residual, self.projection_forward_cache)
            

        # Slice back to correct output dimension if padding occurred
        if self.residual:
            return output[..., :self.out_features]
            
        return output
        
class LinearFrozen(ProjectionModule):
    """Projection-only linear layer with frozen input A.
    Only the weights (B) are updated to satisfy A @ B = Z.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        g: float = 1.0,
        omega: float = 1.0,
        residual: bool = False,
    ):
        super().__init__(outputs=1)
        self.in_features = in_features
        self.out_features = out_features
        self.g = g
        self.omega = omega
        self.use_bias = bias
        self.residual = residual

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(1, out_features))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='linear')
        
        if self.residual:
            with torch.no_grad():
                identity = torch.eye(
                    out_features, in_features, 
                    device=self.weight.device, 
                    dtype=self.weight.dtype
                )
                self.weight.copy_(identity - self.weight)

    def forward(self, input):
        if not config.use_projections:
            b = self.bias.squeeze(0) if self.use_bias else None
            output = F.linear(input, self.weight, b)
            return output / self.omega

        projected_input = input
        weight_matrix = self.weight.T

        if self.use_bias:
            projected_input = F.pad(input, (0, 1), value=1.0)
            weight_matrix = torch.cat([weight_matrix, self.bias], dim=0)

        if self.residual:
            in_dim = projected_input.shape[-1]
            out_dim = weight_matrix.shape[-1]
            padded_dim = max(in_dim, out_dim)

            if padded_dim > in_dim:
                projected_input = F.pad(projected_input, (0, padded_dim - in_dim), value=0.0)

            if padded_dim != in_dim or padded_dim != out_dim:
                padded_weight = weight_matrix.new_zeros((padded_dim, padded_dim))
                padded_weight[:in_dim, :out_dim] = weight_matrix
                weight_matrix = padded_weight

        output = MatMulProjectionFrozenA.apply(
            projected_input, weight_matrix,
            self.g * config.projection_g, self.omega,
            self.residual, self.projection_forward_cache
        )

        if self.residual:
            return output[..., :self.out_features]
            
        return output
        
class ReLU(ProjectionModule):
    def __init__(self, norm="l2"):
        super().__init__(1)
        self.norm = norm

    def forward(self, input, norm=None):
        norm = norm or self.norm
        if config.use_projections:
            if norm == 'linf':
                return ReLULInfinityProjection.apply(input)
            return ReLUProjection.apply(input,  self.projection_forward_cache)
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

class CrossEntropy(ProjectionModule):
    def __init__(self):
        super().__init__(0)

    def forward(self, input, data_target):
        return CrossEntropyProjection.apply(input, data_target)
class Gap(nn.Module):
    """Gap activation function.
    
    The forward pass is the identity (pass-through). The projection
    backward enforces that outputs must lie outside the gap, i.e.
    |y| >= delta/2, forcing the network to commit to a definitive
    decision rather than hovering near zero.

    Args:
        delta: Width of the gap region (default 2.0, giving a gap of
               [-1, 1] in output space).
    """
    def __init__(self, delta: float = 2.0):
        super().__init__()
        self.delta = delta
        
    def forward(self, input):
        if config.use_projections:
            return GapProjection.apply(input, self.delta)
        return input

class GappedStep(nn.Module):
    """Step activation function with a dead zone.
    
    Like the regular Step activation but with a gap of width `delta`
    centred at the origin where the function is undefined:

        f(x) = +1   if x >= delta/2
        f(x) = -1   if x <= -delta/2
        f(x) =  0   otherwise (undefined region, outputs 0)

    Args:
        delta: Width of the gap region (default 2.0).
    """
    def __init__(self, delta: float = 2.0):
        super().__init__()
        self.delta = delta
        
    def forward(self, input):
        if config.use_projections:
            return GappedStepProjection.apply(input, self.delta)
        half = self.delta / 2.0
        return torch.where(input >= half, torch.tensor(1.0, dtype=input.dtype, device=input.device),
               torch.where(input <= -half, torch.tensor(-1.0, dtype=input.dtype, device=input.device),
                            torch.tensor(0.0, dtype=input.dtype, device=input.device)))


class Simplex(nn.Module):
    """Simplex activation function."""
    def __init__(self):
        super().__init__()
        
    def forward(self, input):
        if config.use_projections:
            return SimplexProjection.apply(input)
        import warnings
        warnings.warn("Simplex is not supported in gradient mode")
        return simplex_op_pt(input)

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


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        if config.use_projections:
            return CrossEntropyProjection.apply(input, target)
        return F.cross_entropy(input, target)

class HardMarginLoss(ProjectionModule):
    def __init__(self, delta=1.0):
        super().__init__(0)
        self.delta = delta

    def forward(self, input, target):
        if config.use_projections:
            return HardMarginProjection.apply(input, target)
        err1 = torch.where(target == 1, torch.where(input < self.delta, (input - self.delta)**2, torch.tensor(0.0, device=input.device)), torch.tensor(0.0, device=input.device))
        err0 = torch.where(target == 0, torch.where(input > 0, input**2, torch.tensor(0.0, device=input.device)), torch.tensor(0.0, device=input.device))
        return (err1 + err0).mean()
    
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if config.use_projections:
            return RMSNormProjection.apply(x, self.weight, self.eps, 5)
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)
