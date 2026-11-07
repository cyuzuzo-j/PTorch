"""Back-compat names: the projection-native dispatch (sparsemax / fused V2 / δ
probes) now lives in modules_experimental.MultiheadAttention."""

from functools import partial

from .modules_experimental import CausalSelfAttention as CausalSelfAttentionNative

pairwise_matmul = partial(CausalSelfAttentionNative._project_pairwise_matmul, None)
