"""Experimental modules restored from the openAI_challenge work.

Originally lived in `experimental_modules.py` (deleted in commit fd03390a2);
this file pulls back the framework-aware multi-head attention layer
associated with the openAI_challenge experiment, switched to use the new
`SoftmaxProjection` (previously `SimplexProjection`).

Caveat: reproduced from the deleted source as a starting point — not
re-verified end-to-end.
"""

from typing import Tuple
from frameworks.ptorch.core.ops import MaskedAddProjection

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import config
from ..core.ops import MatMulProjection, SoftmaxProjection, RMSNormProjection, Conversion as ConversionFn, LogitSoftcapInversion
from .modules import Linear, ProjectionModule

class GQAConsensusProjection(torch.autograd.Function):
    """
    Forward: Expands the KV tensor to match the number of Query heads.
    Backward: Averages the targets from the Q-heads to form a consensus target.
    """
    @staticmethod
    def forward(ctx, x, group_size):
        ctx.group_size = group_size
        # x shape: (bsz, num_kv_heads, seqlen, head_dim)
        return x.unsqueeze(2).expand(-1, -1, group_size, -1, -1)

    @staticmethod
    def backward(ctx, z_target):
        # z_target shape: (bsz, num_kv_heads, group_size, seqlen, head_dim)
        # Average the targets across the group_size dimension to find the consensus!
        return z_target.mean(dim=2), None

class RMSNorm(ProjectionModule):
    def __init__(self, eps: float = 1e-5):
        super().__init__(1)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if config.use_projections:
            return RMSNormProjection.apply(x, self.eps)
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies the rotary embedding to the input tensor."""
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

class RotaryFunction(torch.autograd.Function):
    """
    Custom PTorch Autograd Function for Rotary Embeddings.
    Instead of calculating gradients, the backward pass computes the L2 projection
    onto the RoPE architectural constraint set.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # Calculate the forward state
        z = apply_rotary_emb(x, cos, sin)
        
        # Save the input, output, and rotation matrices for the backward projection
        ctx.save_for_backward(x, z, cos, sin)
        return z

    @staticmethod
    def backward(ctx, z_target: torch.Tensor):
        """
        dz is the residual passed from the layer above (z - z_target).
        We use it to reconstruct the target, compute the projection, and 
        return our own residual to pass to the layer below.
        """
        x, z, cos, sin = ctx.saved_tensors
        
        # 2. Compute R^T * z_target (Inverse rotation negates the sine component)
        # Because the rotation is orthogonal, the projection distance formula simplifies.
        rt_z_target = apply_rotary_emb(z_target, cos, -sin)
        
        # 3. Exact L2 Projection onto the constraint z = R * x
        # The optimal target for the input is the midpoint between the forward input 
        # and the inversely-rotated target output.
        x_bar = (x + rt_z_target) / 2.0
        
        return x_bar, None, None


class Rotary(nn.Module):
    """
    PTorch-compliant Rotary Positional Embedding layer.
    """
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
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

    def forward(self, x: torch.Tensor, seq_len: int = None) -> torch.Tensor:
        """
        Applies PTorch-compliant RoPE to the input tensor `x`.
        Expects x shape to be roughly (batch, heads, seq_len, head_dim).
        """
        if seq_len is None:
            seq_len = x.size(-2)
            
        self._update_cache(seq_len, x.device, x.dtype)
        
        # Use the custom PTorch projection function
        return RotaryFunction.apply(
            x, 
            self._cos_cached.to(dtype=x.dtype), 
            self._sin_cached.to(dtype=x.dtype)
        )

class Softcap(nn.Module):
    def __init__(self, logit_softcap=30.0):
        super().__init__()
        self.logit_softcap = logit_softcap

    def forward(self, x):
        # Apply the custom autograd function
        if config.use_projections:  
            return LogitSoftcapInversion.apply(x, self.logit_softcap)
        return self.logit_softcap * torch.tanh(x / self.logit_softcap)

class CausalSelfAttention(nn.Module):
    """Projection-aware Causal Self Attention with RoPE + GQA + RMSNorm.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        norm: str = "l2",
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
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

        self.rotary = Rotary(self.head_dim, base=rope_base)
        self.q_norm = RMSNorm()
        self.k_norm = RMSNorm()

    def _project_pairwise_matmul(self, left: torch.Tensor, right: torch.Tensor,
                                  omega: float = 1.0) -> torch.Tensor:
        if not config.use_projections:
            return (left @ right) / omega
        return MatMulProjection.apply(
            left, right, 5,
            config.projection_alpha,
            config.projection_g,
            omega, None, True, None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, dim = x.shape
        
        if config.use_projections:
            from frameworks.ptorch.core.ops import BranchProjection
            x_branched = BranchProjection.apply(x, 3)
        else:
            x_branched = x

        q = self.c_q(x_branched).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x_branched).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x_branched).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = self.rotary(q, seqlen)
        k = self.rotary(k, seqlen)

        if self.num_heads != self.num_kv_heads:
            group_size = self.num_heads // self.num_kv_heads
            if config.use_projections:
                k = GQAConsensusProjection.apply(k, group_size).reshape(bsz, self.num_heads, seqlen, self.head_dim)
                v = GQAConsensusProjection.apply(v, group_size).reshape(bsz, self.num_heads, seqlen, self.head_dim)
            else:
                k = k.unsqueeze(2).expand(-1, -1, group_size, -1, -1).reshape(bsz, self.num_heads, seqlen, self.head_dim)
                v = v.unsqueeze(2).expand(-1, -1, group_size, -1, -1).reshape(bsz, self.num_heads, seqlen, self.head_dim)

        scale = (self.head_dim ** 0.5)
        qk = self._project_pairwise_matmul(q, k.transpose(-2, -1), omega=scale)

        causal_mask = torch.triu(
            torch.full((seqlen, seqlen), float("-inf"), device=x.device, dtype=qk.dtype),
            diagonal=1,
        )
        if config.use_projections:
            qk = MaskedAddProjection.apply(qk, causal_mask[None, None, :, :])
            attention_weights = SoftmaxProjection.apply(qk, None)
        else:
            qk = qk + causal_mask[None, None, :, :]
            attention_weights = F.softmax(qk, dim=-1)

        y = self._project_pairwise_matmul(attention_weights, v)
        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


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
