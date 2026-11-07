from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.ops import * 
from ..config import config

class ProjectionModule(nn.Module):
    """Base class for projection-aware modules."""
    def __init__(self, outputs=0):
        super().__init__()
        self.outputs = outputs

    def __call__(self, *args, **kwargs):
        # A Parameter fed in as an input must end up with a residual in .grad.
        if config.use_projections:
            args = tuple(map(parameter_operand, args))
            kwargs = {k: parameter_operand(v) for k, v in kwargs.items()}
        return super().__call__(*args, **kwargs)

class Linear(ProjectionModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        alpha: float = 1.0,
        g: float = 1.0,
        omega: float = 1.0,
        num_iters: Optional[int] = None,
        norm = 'l2',
    ):
        super().__init__(outputs=1)
        self.in_features = in_features
        self.out_features = out_features
        # Per-layer multipliers; the effective alpha/g (properties below) are
        # read against the CURRENT config.projection_alpha/projection_g on every
        # forward, so config.update / reset() reach Linear cards exactly like
        # they reach the attention pairwise solves.
        self.alpha_mult = alpha
        self.g_mult = g
        self.omega = omega
        # None = per-norm default: 1 Newton step for l2, config.linf_num_iters for linf.
        self.num_iters = None if num_iters is None else int(num_iters)
        self.use_bias = bias
        self.norm = norm
        self.linf_cache: dict = {}  # warm-start radius of the l∞ solve (config.linf_warm_start)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(1, out_features))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_normal_(self.weight)

    @property
    def alpha(self) -> float:
        return self.alpha_mult * config.projection_alpha

    @property
    def g(self) -> float:
        return self.g_mult * config.projection_g

    def forward(self, input, norm=None):
        norm_type = norm if norm is not None else self.norm

        if not config.use_projections:
            # gradient path
            b = self.bias.squeeze(0) if self.use_bias else None
            output = F.linear(input, self.weight, b)
            return output / self.omega

        # The bias joins the solve as a ones column of the input (folded in by the backward).
        if norm_type == 'linf':
            steps = config.linf_num_iters if self.num_iters is None else self.num_iters
            return MatMulProjectionLinf.apply(input, self.weight.T, steps, self.g, self.omega, self.bias,
                                              self.linf_cache if config.linf_warm_start else None)
        steps = 1 if self.num_iters is None else self.num_iters
        return MatMulProjection.apply(input, self.weight.T, steps, self.alpha, self.g, self.omega,
                                      False, False, self.bias)

class ReLU(ProjectionModule):
    def __init__(self, norm="l2"):
        super().__init__(1)
        self.norm = norm

    def forward(self, input, norm=None):
        norm = norm or self.norm
        if config.use_projections:
            if norm == 'linf':
                return ReLULInfinityProjection.apply(input)
            return ReLUProjection.apply(input)
        return F.relu(input)

class Softmax(ProjectionModule):
    """Projection-aware softmax over the last dimension.

    Forward: z = softmax(input, dim=-1)
    Backward: routes through SoftmaxProjection (mixed L2/KL geometry).
    """
    def __init__(self):
        super().__init__(outputs=1)

    def forward(self, input):
        if config.use_projections:
            return SoftmaxProjection.apply(input)
        return F.softmax(input, dim=-1)


class LeakyReLU(nn.LeakyReLU):
    def __init__(self, negative_slope: float = 0.01, inplace: bool = False,
                 norm: str = 'l2'):
        super().__init__(negative_slope=negative_slope, inplace=inplace)
        self.norm = norm

    def forward(self, input):
        if config.use_projections:
            if self.norm == 'linf':
                return LeakyReLULInfinityProjection.apply(
                    input, self.negative_slope)
            return LeakyReLUProjection.apply(input, self.negative_slope)
        return super().forward(input)


class Step(ProjectionModule):
    """Step activation function."""
    def __init__(self):
        super().__init__()

    def forward(self, input):
        if config.use_projections:
            return StepProjection.apply(input)
        return torch.where(input >= 0, 1.0, -1.0).to(input.dtype)

class QuantizedRelu(ProjectionModule):
    """Quantized ReLU activation.

    Forward:  f(x) = step * round(max(0, x) / step)
    Backward: routes through QuantizeReLUProjection (non-differentiable rung
    projection onto the nearest staircase segment).
    """
    def __init__(self, step: float = 1.0):
        super().__init__()
        self.step = step

    def forward(self, input):
        if config.use_projections:
            return QuantizeReLUProjection.apply(input, self.step)
        return self.step * torch.round(torch.clamp(input, min=0.0) / self.step)

class GappedStep(ProjectionModule):
    """Step activation function with a dead zone.
    
    Like the regular Step activation but with a gap of width `delta`
    centred at the origin where the function is undefined:

        backward pass:
        f(x) = +1   if x >= delta/2
        f(x) = -1   if x <= -delta/2

        forward pass (both modes; gradient mode has zero gradient)
        f(x) = +1   if x >= 0
        f(x) = -1   if x < 0


    Args:
        delta: Width of the gap region (default 2.0).
    """
    def __init__(self, delta: float = 2.0):
        super().__init__()
        self.delta = delta
        
    def forward(self, input):
        if config.use_projections:
            return GappedStepProjection.apply(input, self.delta)
        return torch.where(input >= 0, 1.0, -1.0).to(input.dtype)


class CrossEntropy(ProjectionModule):
    def __init__(self):
        super().__init__(0)

    def forward(self, input, target):
        if config.use_projections:
            return CrossEntropyProjection.apply(
                input, target,
                config.cross_entropy_num_steps,
                config.cross_entropy_lambda,
            )
        return F.cross_entropy(input, target)

class HardMarginLoss(ProjectionModule):
    def __init__(self, delta=1.0):
        super().__init__(0)
        self.delta = delta

    def forward(self, input, target):
        if config.use_projections:
            return HardMarginProjection.apply(input, target * self.delta)  # margin delta on positives
        m = target * self.delta
        return torch.where(m > 0, m - input, input).clamp(min=0).square().mean()


class ProximalHingeMarginLoss(ProjectionModule):
    """Module wrapper for the ProximalHingeMargin autograd Function.

    Uses a soft hinge proximal operator instead of hard-clipping logits
    to the margin boundary.
    """
    def __init__(self, lambda_val=1.0):
        super().__init__(0)
        self.lambda_val = lambda_val

    def forward(self, input, target):
        if config.use_projections:
            return ProximalHingeMargin.apply(input, target, self.lambda_val)
        # Fallback: same hinge-style loss as HardMarginLoss
        return torch.where(target > 0, target - input, input).clamp(min=0).square().mean()
    

class Conv2D(ProjectionModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int], str] = 0,
        bias: bool = True,
        alpha: float = 1.0,
        g: float = 1.0,
        num_iters: int = 5,
    ):
        super().__init__(outputs=1)
        
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        
        kH, kW = self.kernel_size
        if isinstance(padding, str):  # 'same' = ceil(H/stride) outputs for odd k (exact at stride 1)
            padding = {'same': ((kH - 1) // 2, (kW - 1) // 2), 'valid': 0}[padding.lower()]
        self.padding = (padding, padding) if isinstance(padding, int) else tuple(padding)

        # The weights use a "flat" global consensus across both batch and spatial
        # dimensions (consensus_math.md): the patch matrix goes through one Linear solve.
        self.linear = Linear(in_features=in_channels * kH * kW, out_features=out_channels,
                             bias=bias, alpha=alpha, g=g, num_iters=num_iters, norm="2")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not config.use_projections:
            w = self.linear.weight.view(self.linear.out_features, -1, *self.kernel_size)
            b = self.linear.bias.squeeze(0) if self.linear.use_bias else None
            return F.conv2d(input, w, b, self.stride, self.padding)
        lin = self.linear
        if plain_solve():
            return Conv2DProjection.apply(input, lin.weight, lin.bias, self.kernel_size, self.stride, self.padding,
                                          1 if lin.num_iters is None else lin.num_iters, lin.alpha, lin.g)
        patches = ConvPatchProjection.apply(input, self.kernel_size, self.stride, self.padding)
        return self.linear(patches).permute(0, 3, 1, 2)


class MaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        if config.use_projections:
            return MaxPool2DProjection.apply(x, self.kernel_size, self.stride, self.padding)
        return F.max_pool2d(x, self.kernel_size, self.stride, self.padding)
