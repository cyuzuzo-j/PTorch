import torch
import torch.distributed as dist
from .config import config

class ProjectionSGD(torch.optim.SGD):
    """
    Projection-based optimizer wrapped around SGD.
    Converts projection targets into pseudo-gradients (g = p - p_proj).
    Supports all SGD features including momentum and weight decay.
    
    Default learning rate is 1.0, which recovers exact alternating projections
    when no momentum or weight decay is used.
    """
    def __init__(self, params, lr=1.0, **kwargs):
        super().__init__(params, lr=lr, **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        # Convert projection targets (stored in p.grad) to pseudo-gradients
        if config.use_projections:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        # p.grad currently holds the projection target p_proj
                        # We want the gradient g to be p - p_proj
                        p.grad.copy_(p.data - p.grad)
        
        # Now apply the standard SGD step using the pseudo-gradients
        return super().step(closure)

class ProjectionAdam(torch.optim.Adam):
    """
    Projection-based optimizer wrapped around Adam.
    Converts projection targets into pseudo-gradients (g = p - p_proj).
    
    Note: The default learning rate of Adam is 1e-3. You may want to increase this 
    (e.g., to 1.0) since the pseudo-gradients are roughly proportional to parameter values.
    """
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        # Convert projection targets (stored in p.grad) to pseudo-gradients
        if config.use_projections:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        # p.grad currently holds the projection target p_proj
                        p.grad.copy_(p.data - p.grad)
        
        # Now apply the standard Adam step using the pseudo-gradients
        return super().step(closure)

class ProjectionAdagrad(torch.optim.Adagrad):
    """
    Projection-based optimizer wrapped around Adagrad.
    Converts projection targets into pseudo-gradients (g = p - p_proj).
    """
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        # Convert projection targets (stored in p.grad) to pseudo-gradients
        if config.use_projections:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.copy_(p.data - p.grad)
        
        return super().step(closure)

class ProjectionAdadelta(torch.optim.Adadelta):
    """
    Projection-based optimizer wrapped around Adadelta.
    Converts projection targets into pseudo-gradients (g = p - p_proj).
    """
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        # Convert projection targets (stored in p.grad) to pseudo-gradients
        if config.use_projections:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.copy_(p.data - p.grad)
        
        return super().step(closure)


class ProjectionMuon(torch.optim.Muon):
    """
    Projection-based optimizer wrapped around Adadelta.
    Converts projection targets into pseudo-gradients (g = p - p_proj).
    """
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        # Convert projection targets (stored in p.grad) to pseudo-gradients
        if config.use_projections:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.copy_(p.data - p.grad)
        
        return super().step(closure)

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    Muon uses this to normalize matrix-shaped gradients before applying them.
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(dtype=torch.bfloat16)
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X

_POLAR_COEFFS = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375)
]

_POLAR_COEFFS = [
    (a / 1.01, b / 1.01**3, c / 1.01**5) for a, b, c in _POLAR_COEFFS[:-1]
] + [_POLAR_COEFFS[-1]]

@torch.compile()
def zeropower_via_polarexpress(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Computes the polar factor using the optimal Polar Express polynomial method.
    Acts as a drop-in replacement for the static Newton-Schulz5 / Jordan method.
    """
    is_1d = G.ndim == 1
    if is_1d:
        G = G.view(1, -1)
        
    X = G.bfloat16()
    
    # Direction-preserving normalization: floor the *denominator* at eps rather
    # than adding eps. With the additive form a tiny-norm G (e.g. a body
    # activation residual ~1e-7) is scaled toward zero (G/(‖G‖+1e-2) ≈ 100·G),
    # collapsing the polar iteration. clamp_min keeps the unit direction for any
    # ‖G‖ > eps, so small-but-real targets survive. (For weight-Muon, ‖G‖≫eps so
    # this is behaviourally identical to the old additive form.)
    X = X / X.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)
    
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
        
    hs = _POLAR_COEFFS[:steps]
    if steps > len(_POLAR_COEFFS):
        hs += list(repeat(_POLAR_COEFFS[-1], steps - len(_POLAR_COEFFS)))
        
    for a, b, c in hs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
        
    out = X.mT if transposed else X
    
    if is_1d:
        out = out.view(-1)
        
    return out.to(G.dtype)


class ProjectionMuonV2(torch.optim.Optimizer):
    """
    Advanced Projection-based Muon optimizer.
    Converts projection targets into pseudo-gradients (g = p - p_proj) 
    and applies internal Newton-Schulz orthogonalization across distributed ranks.
    """
    def __init__(self, params, lr: float = 1e-3, momentum: float = 0.95, backend_steps: int = 5, nesterov: bool = True,
                 weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 1. Convert projection targets (stored in p.grad) to pseudo-gradients
        if config.use_projections:
            use_hybrid = getattr(config, "use_hybrid", False)
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        is_target = getattr(p, "_is_target", not use_hybrid)
                        if is_target:
                            p.grad.copy_(p.data - p.grad)

        # 2. Setup distributed constants
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        # 3. Apply Muon update rules
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
                
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            weight_decay = group.get("weight_decay", 0.0)

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            # Shard the orthogonalization work across available ranks
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                        
                    g = zeropower_via_polarexpress(g, steps=backend_steps)
                    # Scale correction from Muon reference implementations
                    #g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            # Synchronize updates across all distributed workers
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # 4. Apply the synchronized updates to parameters
            curr = 0
            for p in params:
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                if weight_decay != 0.0:
                    # Decoupled (AdamW/Muon-style) decay. Also bounds the
                    # ||row(A)||-growth feedback that the rel_row activation
                    # rescale otherwise amplifies late in training.
                    p.mul_(1.0 - lr * weight_decay)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss