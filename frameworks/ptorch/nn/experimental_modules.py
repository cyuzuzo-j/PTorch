import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from ..core.ops import *
from .. import config
from .modules import Linear, RMSNorm, Conversion as ConversionFn

def max_proj_pt_batch(a, z):
    """
    Vectorized projection onto the maximum function graph.
    Processes a batch of arrays simultaneously.
    
    Args:
        a: Tensor of shape (B, P) where P is the patch size (e.g., kernel_h * kernel_w).
        z: Tensor of shape (B, 1) containing the target maximums.
    Returns:
        Projected tensor of shape (B, P).
    """
    B, P = a.shape
    
    # 1. Sort arrays
    a_sorted, idx = torch.sort(a, dim=1)

    # 2. Compute candidate maxima (z_k)
    a_sorted_flipped = torch.flip(a_sorted, dims=[1])
    cumsum_flipped = torch.cumsum(a_sorted_flipped, dim=1)
    divisors = torch.arange(2, P + 2, device=a.device, dtype=a.dtype).unsqueeze(0)
    
    z_k_flipped = (cumsum_flipped + z) / divisors
    z_k = torch.flip(z_k_flipped, dims=[1])

    # 3. Compute candidate arrays (a_k)
    # Create upper triangular mask (1, P, P)
    i_ge_k = torch.triu(torch.ones((P, P), dtype=torch.bool, device=a.device)).unsqueeze(0)
    
    # a_k shape: (B, P_candidate, P_element)
    a_k = torch.where(i_ge_k, z_k.unsqueeze(2), a_sorted.unsqueeze(1))

    # 4. Compute distances
    dist = ((a_k - a_sorted.unsqueeze(1)) ** 2).sum(dim=2) + (z_k - z) ** 2

    # 5. Select valid candidates
    # Add a small epsilon (1e-5) to handle floating point inaccuracies
    valid = torch.max(a_k, dim=2)[0] <= z_k + 1e-5
    dist_valid = torch.where(valid, dist, torch.tensor(float('inf'), device=a.device, dtype=dist.dtype))

    # 6. Select candidate minimizing distance
    k = torch.argmin(dist_valid, dim=1)

    # Extract the best sorted candidate array for each batch item
    batch_indices = torch.arange(B, device=a.device)
    best_a_k_sorted = a_k[batch_indices, k, :]

    # 7. Unsort back to original spatial layout
    a_proj = torch.zeros_like(a)
    a_proj.scatter_(1, idx, best_a_k_sorted)

    return a_proj


class MaxPool2DProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, kernel_size, stride=None, padding=0):
        if stride is None:
            stride = kernel_size
        
        # PJAX enforces non-overlapping windows for projection validity
        k_tuple = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        s_tuple = (stride, stride) if isinstance(stride, int) else tuple(stride)
        
        if k_tuple != s_tuple:
            raise ValueError("MaxPool projection requires strides == pool_size")

        ctx.save_for_backward(input)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        
        return F.max_pool2d(input, kernel_size, stride, padding)

    @staticmethod
    def backward(ctx, z_target):
        input, = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        padding = ctx.padding

        N, C, H_in, W_in = input.shape
        _, _, H_out, W_out = z_target.shape

        kH = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
        kW = kernel_size[1] if isinstance(kernel_size, tuple) else kernel_size

        # 1. Extract patches using unfold
        # Shape: (N, C * kH * kW, H_out * W_out)
        patches = F.unfold(input, kernel_size, stride=stride, padding=padding)
        L = patches.shape[-1] # Number of spatial patches (H_out * W_out)
        
        # 2. Reshape to isolate each local pooling window
        # (N, C, kH * kW, L) -> Permute to (N, C, L, kH * kW)
        patches = patches.view(N, C, kH * kW, L).permute(0, 1, 3, 2).contiguous()
        
        # Flatten batch, channels, and spatial dims into a single batch dimension
        a_batch = patches.view(-1, kH * kW) # Shape: (B, P)
        z_batch = z_target.reshape(-1, 1)      # Shape: (B, 1)

        # 3. Apply vectorized projection
        a_proj_batch = max_proj_pt_batch(a_batch, z_batch)

        # 4. Reconstruct the image
        # Unflatten back to (N, C, L, kH * kW)
        a_proj_patches = a_proj_batch.view(N, C, L, kH * kW)
        
        # Permute to (N, C, kH * kW, L) and collapse C and spatial patch dims
        a_proj_unfolded = a_proj_patches.permute(0, 1, 3, 2).contiguous().view(N, C * kH * kW, L)

        # F.fold restores the image geometry. Because stride == kernel_size, there's no overlap.
        a_proj = F.fold(
            a_proj_unfolded, 
            output_size=(H_in, W_in), 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding
        )

        
        return a_proj, None, None, None

# Module Wrapper
class MaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        return MaxPool2DProjection.apply(x, self.kernel_size, self.stride, self.padding)
class LinearOrth(nn.Module):
    """Simplified linear layer using OrthogonalRotationProjection.
    
    This layer maintains a square weight matrix that is initialized to be 
    orthogonal and updated using the closed-form Orthogonal Procrustes projection.
    """
    def __init__(self, dim: int, alpha: float = 1.0, gamma: float = 1.0):
        super().__init__()
        self.dim = dim
        self.alpha = alpha
        self.gamma = gamma
        self.weight = nn.Parameter(torch.empty(dim, dim))
        nn.init.orthogonal_(self.weight)
        
        # Ensure it starts as a pure rotation (det=1)
        with torch.no_grad():
            if torch.linalg.det(self.weight) < 0:
                self.weight[0] *= -1

    def forward(self, x):
        if not config.use_projections:
            # Standard path: y = x @ W.T
            return F.linear(x, self.weight)
        
        # Projection path: y = R @ x (where R is the weight)
        return OrthogonalRotationProjection.apply(self.weight, x, self.alpha, self.gamma)


class LinearHybrid(nn.Module):
    """Hybrid linear layer: real gradients upstream + standard weight gradient for ProjectionMuon.

    This layer is the drop-in replacement for ``Linear`` when using the hybrid
    gradient-projection approach:

    * **Upstream**: returns the standard chain-rule gradient ``∂L/∂A = Z_grad @ B^T``.
      Signal propagates through all layers at full magnitude — no vanishing-target problem.

    * **Weights**: returns the standard gradient ``A^T @ Z_grad`` in ``p.grad``.
      Feed parameters to ``ProjectionMuon`` (already in optim_static.py); Muon's
      Newton-Schulz step auto-scales and orthogonalises the gradient before updating.

    Args:
        in_features:  Number of input features.
        out_features: Number of output features.
        bias:         If True, appends a bias via input augmentation (default True).
        omega:        Output scale divisor, mirrors the ``Linear`` convention (default 1.0).
        norm:         Projection norm — 'l2', 'linf', 'l1', or 'l<p>' (default 'l2').
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        omega: float = 1.0,
        eta: float = 0.01,
        norm = 'l2',
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.omega = omega
        self.use_bias = bias
        self.eta = eta
        self.norm = norm

        # Separate parameters to ensure optimizers (e.g., AdamW) can exclude bias from weight decay
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(1, out_features))
        else:
            self.register_parameter('bias', None)

        # 1. Correct Initialization Variance:
        # Now matches nn.Linear shape (out_features, in_features) natively
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, input):
        # Speed/Memory Efficiency: Use F.pad instead of torch.cat for input bias augmentation
        projected_input = input
        weight_matrix = self.weight.T

        if self.use_bias:
            projected_input = F.pad(input, (0, 1), value=1.0)
            weight_matrix = torch.cat([weight_matrix, self.bias], dim=0)

        p_value = 2.0 if self.norm == 'l2' else float('inf')

        return MatMulProjectionHybrid.apply(
            projected_input,
            weight_matrix,
            self.omega,
            self.eta,
            False,  # residual
            1,      # num_steps
            1.0,    # alpha
            1.0,    # g
            None,   # proj_cache
            p_value, # p (norm exponent)
        )

class LinearAbs(nn.Module):
    """Fused Linear+Abs layer using exact ℓ∞ projection.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        omega: float = 1.0,
        num_iters: int = 5,
        gamma: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.omega = omega
        self.use_bias = bias
        self.num_iters = num_iters
        self.gamma = gamma
        self.proj_cache = {}

        # Separate parameters to ensure optimizers (e.g., AdamW) can exclude bias from weight decay
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(1, out_features))
        else:
            self.register_parameter('bias', None)

        # 1. Correct Initialization Variance:
        # Now matches nn.Linear shape (out_features, in_features) natively
        nn.init.kaiming_normal_(self.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, input):
        if not config.use_projections:
            b = self.bias.squeeze(0) if self.use_bias else None
            z = F.linear(input, self.weight, b)
            return torch.abs(z) / self.omega

        # Speed/Memory Efficiency: Use F.pad instead of torch.cat for input bias augmentation
        projected_input = input
        weight_matrix = self.weight.T

        if self.use_bias:
            projected_input = F.pad(input, (0, 1), value=1.0)
            weight_matrix = torch.cat([weight_matrix, self.bias], dim=0)

        output = AbsBilinearProjectionLinf.apply(
            projected_input,
            weight_matrix,
            self.num_iters,
            self.gamma,
            self.omega,
            self.proj_cache,
        )
        return output

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

class ReLUHybrid(nn.Module):
    """
    Standard gradient-backed ReLU for use in Hybrid networks.
    Unaffected by the global `use_projections` state, allowing real gradients
    to flow cleanly when the model is trained with hybrid projections.
    """
    def __init__(self, **kwargs):
        super().__init__()
        # kwargs (like norm) are ignored since Hybrid always propagates gradients natively

    def forward(self, input):
        # Natively uses fundamental PyTorch autograd gradients.
        return F.relu(input)



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



class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional Grouped Query Attention (GQA).

    Implements the multi-head attention mechanism as described in "Attention
    Is All You Need" (Vaswani et al., 2017), extended with GQA support from
    "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head
    Checkpoints" (Ainslie et al., 2023).

    When ``num_kv_heads < heads``, key/value projections use fewer heads
    (reducing the KV cache and parameter count). The KV heads are repeated
    to match the query head count during attention computation.

    Uses SimplexProjection instead of softmax for attention weights,
    and MatMulExactProjection for projected matrix multiplications.

    Args:
        model_features: dimensionality of input and output features.
        qkv_features: dimensionality of query, key, and value vectors per head.
        heads: number of query attention heads.
        num_kv_heads: number of key/value heads (defaults to ``heads`` for
            standard MHA). Must evenly divide ``heads``.

    Attributes:
        query_layer: linear layer for computing queries.
        key_layer: linear layer for computing keys.
        value_layer: linear layer for computing values.
        out_layer: output projection layer.
        heads: number of query attention heads.
        num_kv_heads: number of key/value attention heads.
    """

    def __init__(self, model_features, qkv_features, heads, num_kv_heads=None,
                 alpha=1.0, g=1.0, attention_type='simplex', norm="l2",
                 use_rotary=False, rope_base=10000.0):
        super().__init__()
        if num_kv_heads is None:
            num_kv_heads = heads
        if heads % num_kv_heads != 0:
            raise ValueError(
                f"heads ({heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )
        self.heads = heads
        self.num_kv_heads = num_kv_heads
        self.kv_group_size = heads // num_kv_heads  # how many Q heads per KV head
        self.attention_type = attention_type

        kv_dim = num_kv_heads * qkv_features
        self.query_layer = Linear(model_features, heads * qkv_features, alpha=alpha, g=g, norm=norm)
        self.key_layer = Linear(model_features, kv_dim, alpha=alpha, g=g, norm=norm)
        self.value_layer = Linear(model_features, kv_dim, alpha=alpha, g=g, norm=norm)
        self.out_layer = Linear(heads * qkv_features, model_features, alpha=alpha, g=g, norm=norm)

        self.use_rotary = use_rotary
        if use_rotary:
            self.rotary = Rotary(qkv_features, base=rope_base)

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
            self.query_layer.alpha * config.projection_alpha,
            1 * config.projection_g,
            omega,
            None,
            True,
            False,
        )

    def _expand_kv_heads(self, x):
        """Repeat KV heads to match the number of query heads.

        Args:
            x: tensor of shape ``(batch, num_kv_heads, seq_len, head_dim)``.

        Returns:
            tensor of shape ``(batch, heads, seq_len, head_dim)``.
        """
        if self.kv_group_size == 1:
            return x
        # (B, num_kv_heads, S, D) -> (B, num_kv_heads, 1, S, D) -> (B, num_kv_heads, G, S, D)
        return (
            x.unsqueeze(2)
             .expand(-1, -1, self.kv_group_size, -1, -1)
             .reshape(x.shape[0], self.heads, x.shape[2], x.shape[3])
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

        batch_size, seq_len = input.shape[:2]

        # Q: (B, S, heads*D) -> (B, heads, S, D)
        q = q.reshape(batch_size, seq_len, self.heads, -1).permute(0, 2, 1, 3)

        # K, V: (B, S, num_kv_heads*D) -> (B, num_kv_heads, S, D)
        head_dim = q.shape[-1]
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, head_dim).permute(0, 2, 1, 3)

        if getattr(self, 'use_rotary', False):
            cos, sin = self.rotary(seq_len, q.device, q.dtype)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        # Expand KV heads to match Q heads for attention: (B, heads, S, D)
        k = self._expand_kv_heads(k)
        v = self._expand_kv_heads(v)

        # Compute attention scores
        scale = (head_dim ** 0.5)
        qk = self._project_pairwise_matmul(q, k.transpose(-2, -1), omega=scale)
        if self.attention_type == 'hardmax':
            if config.use_projections:
                qk = HardmaxProjection.apply(qk)
            else:
                qk = hardmax_op_pt(qk)
        elif self.attention_type == 'simplex':
            if config.use_projections:
                qk = SimplexProjection.apply(qk)
            else:
                qk = simplex_op_pt(qk)
        else:
            if config.use_projections:
                qk = SoftmaxProjection.apply(qk)
            else:
                qk = F.softmax(qk, dim=-1)
        attention_weights = qk

        # Weighted sum of values
        o = self._project_pairwise_matmul(qk, v)

        # Merge heads: (B, heads, S, D) -> (B, S, heads*D)
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

class ReLUSquared(nn.Module):
    """
    Projection-aware Squared ReLU.
    Forward: returns max(0, x)^2
    Backward: exact Euclidean projection onto the Squared ReLU constraint graph.
    """
    def __init__(self):
        super().__init__()

    def forward(self, input):
        if config.use_projections:
            return SquaredReLUProjection.apply(input)
        return torch.square(torch.relu(input))
    

class Quantize(nn.Module):
    """Infinite-ladder quantization activation.

    Rounds each element to the nearest multiple of ``step``, producing
    a staircase function with uniform step height extending to ±∞:

        f(x) = step × round(x / step)

    Unlike a bounded quantizer (which clamps to a finite set of levels),
    this function has infinitely many rungs so that every real-valued
    input maps to a well-defined level without range saturation.

    In projection mode the backward pass computes the exact Euclidean
    projection of ``(x, z_target)`` onto the graph of the staircase,
    giving a meaningful geometric training signal despite the piecewise-
    constant forward pass.

    Args:
        step: Distance between adjacent quantization levels (default 1.0).
    """
    def __init__(self, step: float = 0.1):
        super().__init__()
        if step <= 0:
            raise ValueError(f"step must be positive, got {step}")
        self.step = step

    def forward(self, input):
        if config.use_projections:
            return QuantizeReLUProjection.apply(input, self.step)
        return self.step * torch.round(input / self.step)
    


class Rotary(nn.Module):
    # Caches cos/sin tables per sequence length on the current device.
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    """
    Projection-aware Causal Self Attention featuring:
    - RoPE
    - RMSNorm
    - GQA (Grouped Query Attention)
    - Causal Masking
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        norm="l2"
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        
        self.c_q = Linear(dim, dim, bias=False, norm=norm)
        self.c_k = Linear(dim, kv_dim, bias=False, norm=norm)
        self.c_v = Linear(dim, kv_dim, bias=False, norm=norm)
        self.proj = Linear(dim, dim, bias=False, norm=norm)
        
        self.q_gain = nn.Parameter(torch.full((num_heads, 1), qk_gain_init, dtype=torch.float32))
        self.rotary = Rotary(self.head_dim, base=rope_base)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def _project_pairwise_matmul(self, left, right, omega=1.0):
        if not config.use_projections:
            return (left @ right) / omega
        # Using analytical alpha/g scaling of 1.0 for these internal matrices
        return MatMulProjection.apply(
            left, right, 5, 1.0 * config.projection_alpha, 1.0 * config.projection_g, omega, None, True, False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        
        q = q * self.q_gain.to(dtype=q.dtype).view(1, self.num_heads, 1, 1)
        
        # Expand KV heads to match Q heads for attention: (B, heads, S, D)
        if self.num_heads != self.num_kv_heads:
            group_size = self.num_heads // self.num_kv_heads
            k = k.unsqueeze(2).expand(-1, -1, group_size, -1, -1).reshape(bsz, self.num_heads, seqlen, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, group_size, -1, -1).reshape(bsz, self.num_heads, seqlen, self.head_dim)

        # Compute attention scores
        scale = (self.head_dim ** 0.5)
        qk = self._project_pairwise_matmul(q, k.transpose(-2, -1), omega=scale)
        
        # Causal mask application
        causal_mask = torch.triu(torch.full((seqlen, seqlen), float("-inf"), device=x.device, dtype=x.dtype), diagonal=1)
        qk = qk + causal_mask[None, None, :, :]
        
        if config.use_projections:
            attention_weights = SimplexProjection.apply(qk)
        else:
            attention_weights = F.softmax(qk, dim=-1)
            
        y = self._project_pairwise_matmul(attention_weights, v)
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)