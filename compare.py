import torch
import jax
import jax.numpy as jnp
import numpy as np

# PyTorch version
def simplex_proj_pt(a, z):
    n = a.size(-1)
    w = a + z
    idx = torch.argsort(w, dim=-1)
    
    a_s = torch.gather(a, -1, idx)
    z_s = torch.gather(z, -1, idx)
    w_s = torch.gather(w, -1, idx)
    
    a_d = torch.flip(a_s, dims=[-1])
    z_d = torch.flip(z_s, dims=[-1])
    w_d = torch.flip(w_s, dims=[-1])
    
    cssv = torch.cumsum(w_d, dim=-1)
    k_range = torch.arange(1, n + 1, device=a.device, dtype=a.dtype)
    tau = (cssv - 2.0) / k_range
    
    tau_matrix = tau.unsqueeze(-1).expand(*tau.shape[:-1], n, n)
    w_d_row = w_d.unsqueeze(-2).expand(*w_d.shape[:-1], n, n)
    a_d_row = a_d.unsqueeze(-2).expand(*a_d.shape[:-1], n, n)
    z_d_row = z_d.unsqueeze(-2).expand(*z_d.shape[:-1], n, n)
    
    active_mask = torch.tril(torch.ones((n, n), dtype=torch.bool, device=a.device))
    
    a_active = (w_d_row + tau_matrix) / 2.0
    z_active = (w_d_row - tau_matrix) / 2.0
    
    a_clamped = torch.minimum(a_d_row, tau_matrix)
    z_clamped = torch.zeros_like(z_active)
    
    a_cand = torch.where(active_mask, a_active, a_clamped)
    z_cand = torch.where(active_mask, z_active, z_clamped)
    
    dist = torch.sum((a_cand - a_d_row)**2 + (z_cand - z_d_row)**2, dim=-1)
    
    dist_valid = torch.where(w_d >= tau, dist, torch.tensor(float('inf'), device=a.device, dtype=a.dtype))
    
    best_k = torch.argmin(dist_valid, dim=-1, keepdim=True)
    best_k_expanded = best_k.unsqueeze(-1).expand(*best_k.shape[:-1], 1, n)
    best_a_d = torch.gather(a_cand, -2, best_k_expanded).squeeze(-2)
    
    best_a_s = torch.flip(best_a_d, dims=[-1])
    inverse_idx = torch.argsort(idx, dim=-1)
    best_a_orig = torch.gather(best_a_s, -1, inverse_idx)
    return best_a_orig

# JAX version
@jax.jit
def simplex_proj_jax(a, z):
    n = a.shape[-1]
    w = a + z
    idx = jnp.argsort(w, axis=-1)
    
    a_s = jnp.take_along_axis(a, idx, axis=-1)
    z_s = jnp.take_along_axis(z, idx, axis=-1)
    w_s = jnp.take_along_axis(w, idx, axis=-1)
    
    a_d = jnp.flip(a_s, axis=-1)
    z_d = jnp.flip(z_s, axis=-1)
    w_d = jnp.flip(w_s, axis=-1)
    
    cssv = jnp.cumsum(w_d, axis=-1)
    k_range = jnp.arange(1, n + 1)
    tau = (cssv - 2.0) / k_range
    
    tri = jnp.tril(jnp.ones((n, n), dtype=bool))
    
    tau_m = tau[..., :, jnp.newaxis]
    w_d_m = w_d[..., jnp.newaxis, :]
    a_d_m = a_d[..., jnp.newaxis, :]
    z_d_m = z_d[..., jnp.newaxis, :]
    
    a_active = (w_d_m + tau_m) / 2.0
    z_active = (w_d_m - tau_m) / 2.0
    a_clamped = jnp.minimum(a_d_m, tau_m)
    z_clamped = jnp.zeros_like(z_active)
    
    a_cand = jnp.where(tri, a_active, a_clamped)
    z_cand = jnp.where(tri, z_active, z_clamped)
    
    dist = jnp.sum((a_cand - a_d_m)**2 + (z_cand - z_d_m)**2, axis=-1)
    
    valid = w_d >= tau
    dist_valid = jnp.where(valid, dist, jnp.inf)
    
    best_k = jnp.argmin(dist_valid, axis=-1, keepdims=True)
    best_a_d = jnp.take_along_axis(a_cand, best_k[..., jnp.newaxis], axis=-2).squeeze(-2)
    
    best_a_s = jnp.flip(best_a_d, axis=-1)
    inverse_idx = jnp.argsort(idx, axis=-1)
    best_a_orig = jnp.take_along_axis(best_a_s, inverse_idx, axis=-1)
    
    return best_a_orig

# Test them
for _ in range(100):
    a_np = np.random.randn(2, 5).astype(np.float32)
    z_np = np.random.randn(2, 5).astype(np.float32)
    
    a_pt = torch.tensor(a_np)
    z_pt = torch.tensor(z_np)
    res_pt = simplex_proj_pt(a_pt, z_pt).numpy()
    
    a_jx = jnp.array(a_np)
    z_jx = jnp.array(z_np)
    res_jx = np.array(simplex_proj_jax(a_jx, z_jx))
    
    if not np.allclose(res_pt, res_jx, atol=1e-5):
        print("MISMATCH!")
        print("PT:", res_pt)
        print("JX:", res_jx)
        break
else:
    print("ALL MATCH!")
