from typing import Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.ops import * 
from ..config import config

class ProjectionModule(nn.Module):
    """Base class for projection-aware modules."""
    def __init__(self, outputs):
        super().__init__()
        self.outputs = outputs
        self.projection_forward_cache = [None for _ in range(outputs)]
    
class Linear(ProjectionModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        alpha: float = 1.0,
        g: float = 1.0,
        omega: float = 1.0,
        num_iters: int = 1,
        norm = 'l2',
        use_cache=True
    ):
        super().__init__(outputs=1)
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha*config.projection_alpha
        self.g = g*config.projection_g
        self.omega = omega
        self.num_iters = int(num_iters)
        self.use_bias = bias
        self.norm = norm
        self.proj_cache: dict = {}
        self.use_cache = use_cache

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(1, out_features))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_normal_(self.weight)
        
    def forward(self, input, norm=None):
        norm_type = norm if norm is not None else self.norm

        if not config.use_projections:
            # gradient path
            b = self.bias.squeeze(0) if self.use_bias else None
            output = F.linear(input, self.weight, b)
            return output / self.omega

        projected_input = input
        weight_matrix = self.weight.T
        if not self.use_cache:
            self.proj_cache = {}

        if self.use_bias:
            projected_input = F.pad(input, (0, 1), value=1.0)
            weight_matrix = torch.cat([weight_matrix, self.bias], dim=0)

        # Dispatch to  projection function
        if norm_type == 'linf':
            output = MatMulProjectionLinf.apply(
                projected_input, weight_matrix, self.num_iters,
                self.g, self.omega,
                self.proj_cache,
            )
        else:
            output = MatMulProjection.apply(
                projected_input, weight_matrix, self.num_iters,
                self.alpha ,
                self.g, self.omega,
                self.proj_cache, False, self.projection_forward_cache)
            
        return output

class ReLU(ProjectionModule):
    def __init__(self, norm="l2"):
        super().__init__(1)
        self.norm = norm

    def forward(self, input, norm=None):
        norm = norm or self.norm
        if config.use_projections:
            if norm == 'linf':
                return ReLULInfinityProjection.apply(input,  self.projection_forward_cache)
            return ReLUProjection.apply(input,  self.projection_forward_cache)
        return super().forward(input)

class LeakyReLU(nn.LeakyReLU):
    def __init__(self, negative_slope: float = 0.01, inplace: bool = False):
        super().__init__(negative_slope=negative_slope, inplace=inplace)

    def forward(self, input):
        if config.use_projections:
            return LeakyReLUProjection.apply(input, self.negative_slope)
        return super().forward(input)


class Step(ProjectionModule):
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

class GappedStep(ProjectionModule):
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

class Dropout(ProjectionModule):
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


class CrossEntropy(ProjectionModule):
    def __init__(self):
        super().__init__(0)

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
        err1 = torch.where(target > 0, torch.where(input < target, (input - target)**2, torch.tensor(0.0, device=input.device)), torch.tensor(0.0, device=input.device))
        err0 = torch.where(target <= 0, torch.where(input > 0, input**2, torch.tensor(0.0, device=input.device)), torch.tensor(0.0, device=input.device))
        return (err1 + err0).mean()
    

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
        num_iters: int = 15,
    ):
        super().__init__(outputs=1)
        
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        
        if isinstance(padding, int):
            self.padding_mode = (padding, padding)
        elif isinstance(padding, str):
            self.padding_mode = padding.lower()
            if self.padding_mode not in ['same', 'valid']:
                raise ValueError(f"Unknown padding mode: {padding}")
        else:
            self.padding_mode = tuple(padding)

        kH, kW = self.kernel_size
        
        # As noted in consensus_math.md: The weights utilize a "flat" global consensus 
        # across both batch and spatial dimensions. We leverage the Linear layer so 
        # it natively routes to MatMulProjectionHybrid without O(MNK) expansion issues.
        self.linear = Linear(
            in_features=in_channels * kH * kW, 
            out_features=out_channels, 
            bias=False, 
            alpha=alpha, 
            g=g, 
            num_iters=num_iters,
            norm="2",
            use_cache=False
        )

    def _resolve_padding(self, H: int, W: int) -> Tuple[int, int]:
        """Dynamically computes spatial padding offsets."""
        if self.padding_mode == 'same':
            sH, sW = self.stride
            kH, kW = self.kernel_size
            pad_h = max(0, (H - 1) * sH + kH - H) // 2
            pad_w = max(0, (W - 1) * sW + kW - W) // 2
            return (pad_h, pad_w)
        elif self.padding_mode == 'valid':
            return (0, 0)
        return self.padding_mode

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        padding = self._resolve_padding(input.shape[2], input.shape[3])

        if getattr(config, 'use_projections', False):
            # Enforce Spatial Consensus (Activations)
            patches = ConvPatchProjection.apply(input, self.kernel_size, self.stride, padding)
        else:
            # Standard Unfold Pipeline (Gradients)
            patches = F.unfold(input, self.kernel_size, dilation=1, padding=padding, stride=self.stride)
            kH, kW = self.kernel_size
            sH, sW = self.stride
            pad_h, pad_w = padding
            H_out = (input.shape[2] + 2 * pad_h - kH) // sH + 1
            W_out = (input.shape[3] + 2 * pad_w - kW) // sW + 1
            patches = patches.view(input.shape[0], -1, H_out, W_out).permute(0, 2, 3, 1)

        # Enforce Flat Global Consensus (Weights & Bias) via MatMulProjection
        out = self.linear(patches)
        
        # Reshape back to Image Topology: (N, C_out, H_out, W_out)
        return out.permute(0, 3, 1, 2)


class MaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        return MaxPool2DProjection.apply(x, self.kernel_size, self.stride, self.padding)
