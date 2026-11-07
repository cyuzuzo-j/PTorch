"""Optimizers for projection training. Every parameter's .grad holds the
pseudo-gradient theta - theta_bar (in gradient mode, the true gradient), so these
are the torch optimizers as they are, with projection-appropriate defaults."""
import torch
from .core.ops import _TC_MODE

class _ParametersOnly:
    """A plain leaf tensor is an activation: its .grad holds a target, which must
    not be stepped on as if it were a pseudo-gradient."""
    def add_param_group(self, param_group):
        params = param_group['params']
        params = [params] if isinstance(params, torch.Tensor) else list(params)
        if not all(isinstance(p, torch.nn.Parameter) for p in params):
            raise TypeError(f"{type(self).__name__} only optimizes nn.Parameters")
        super().add_param_group({**param_group, 'params': params})

class ProjectionSGD(_ParametersOnly, torch.optim.SGD):
    """SGD on projection pseudo-gradients. Default lr=1.0 recovers exact
    alternating projections when no momentum or weight decay is used."""
    def __init__(self, params, lr=1.0, **kwargs):
        super().__init__(params, lr=lr, **kwargs)

class ProjectionAdam(_ParametersOnly, torch.optim.Adam):
    """Adam on projection pseudo-gradients. Adam's default lr is 1e-3; you may
    want more, since pseudo-gradients scale with parameter values."""

class ProjectionAdagrad(_ParametersOnly, torch.optim.Adagrad):
    """Adagrad on projection pseudo-gradients."""

class ProjectionAdadelta(_ParametersOnly, torch.optim.Adadelta):
    """Adadelta on projection pseudo-gradients."""

def _is_matrix(p):
    """Muon orthogonalizes matrices; a (1, N) bias row or a vector is not one."""
    return p.ndim == 2 and min(p.shape) > 1


# Compile modes that replay compiled calls as CUDA graphs (cudagraph trees).
_TC_CUDAGRAPHS = _TC_MODE in ("reduce-overhead", "max-autotune")


def _mark_static(t):
    """Under cudagraph trees, let a compiled call mutate / read t in place:
    a mutated non-static input makes inductor skip cudagraphs, and a read-only
    one is copied into the graph's own buffer on every replay. Only for
    allocate-once tensors; parameters are static already. Unguarded: a new
    address (e.g. after load_state_dict) re-records the graph rather than
    recompiling. No-op in the other modes, so the default path is unchanged,
    and while dynamo traces an outer compiled step (fast_step's
    compile_optimizer), which may not call it and marks optimizer state
    static itself."""
    if _TC_CUDAGRAPHS and not torch.compiler.is_compiling():
        torch._dynamo.mark_static_address(t)


@torch.compile(dynamic=False, mode=_TC_MODE)
def _muon_update(params, grads, bufs, lrs, adjusted_lrs, weight_decay, momentum,
                 nesterov: bool, ns_coefficients, ns_steps: int, eps: float):
    """torch.optim.Muon's update (same math as its single-tensor path) for all
    matrices of one param group, as a single compiled call instead of ~30
    eager launches per matrix; dynamo unrolls the loop. lrs / adjusted_lrs are
    0-dim tensors: dynamo guards on Python floats, so a scheduled lr would
    otherwise recompile every step until the cache limit and then fall back to
    eager."""
    a, b, c = ns_coefficients
    for p, grad, buf, lr, adjusted_lr in zip(params, grads, bufs, lrs, adjusted_lrs):
        buf.lerp_(grad, 1 - momentum)
        update = grad.lerp(buf, momentum) if nesterov else buf
        X = update.bfloat16()
        transposed = X.size(0) > X.size(1)
        if transposed:
            X = X.T
        X = X / X.norm().clamp(min=eps)
        for _ in range(ns_steps):
            gram = X @ X.T
            X = torch.addmm(X, torch.addmm(gram, gram, gram, beta=b, alpha=c), X, beta=a)
        if transposed:
            X = X.T
        p.mul_(1 - lr * weight_decay)
        p.sub_(X.to(p.dtype) * adjusted_lr)


class ProjectionMuon(_ParametersOnly, torch.optim.Muon):
    """torch.optim.Muon on projection pseudo-gradients.

    Like the standard Muon recipe, only true matrices get Muon; vectors and
    (1, N) bias rows go to a ProjectionAdam (`adam_lr`, `adam_kwargs`), held in
    `vector_optimizer` and stepped / zeroed / checkpointed together with Muon.
    Orthogonalizing a single row only rescales it to unit norm, and costs a
    full Newton-Schulz launch sequence per bias.

    All matrix updates of a param group run as one compiled call
    (_muon_update); the bf16 Newton-Schulz then rounds slightly differently
    from eager torch.optim.Muon.
    The Adam part uses the fused kernel on CUDA unless `adam_kwargs` says
    otherwise.
    """
    def __init__(self, params, adam_lr=1e-3, adam_kwargs=None, **kwargs):
        params = list(params)
        groups = params if params and isinstance(params[0], dict) else [{"params": params}]
        matrix_groups, vector_params = [], []
        for g in groups:
            ps = list(g["params"])
            matrix_groups.append({**g, "params": [p for p in ps if _is_matrix(p)]})
            vector_params += [p for p in ps if not _is_matrix(p)]
        matrix_groups = [g for g in matrix_groups if g["params"]]
        super().__init__(matrix_groups, **kwargs)
        adam_kwargs = dict(adam_kwargs or {})
        if vector_params and all(p.is_cuda for p in vector_params):
            adam_kwargs.setdefault("fused", True)
        self.vector_optimizer = (ProjectionAdam(vector_params, lr=adam_lr, **adam_kwargs)
                                 if vector_params else None)
        self._lr_tensors = {}  # p -> (lr, adjusted_lr, 0-dim tensor of each)

    def _lr_tensor_pair(self, p, lr, adjusted_lr):
        """Persistent 0-dim lr tensors for p, refilled only when the value
        changes (no per-step host-to-device copy; CUDA-graph capturable). Kept
        out of self.state so checkpoints and state rewinds never touch them."""
        cached = self._lr_tensors.get(p)
        if cached is None:
            t = (torch.tensor(lr, dtype=torch.float32, device=p.device),
                 torch.tensor(adjusted_lr, dtype=torch.float32, device=p.device))
            for x in t:
                _mark_static(x)
            self._lr_tensors[p] = (lr, adjusted_lr) + t
            return t
        old_lr, old_adj, lr_t, adj_t = cached
        if lr != old_lr or adjusted_lr != old_adj:
            lr_t.fill_(lr)
            adj_t.fill_(adjusted_lr)
            self._lr_tensors[p] = (lr, adjusted_lr, lr_t, adj_t)
        return lr_t, adj_t

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            ps, grads, bufs, lrs, adjusted_lrs = [], [], [], [], []
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p.grad, memory_format=torch.preserve_format)
                    _mark_static(state["momentum_buffer"])
                adjusted_lr = torch.optim._muon._adjust_lr(group["lr"], group["adjust_lr_fn"], p.shape)
                lr_t, adjusted_lr_t = self._lr_tensor_pair(p, float(group["lr"]), float(adjusted_lr))
                ps.append(p)
                grads.append(p.grad)
                bufs.append(state["momentum_buffer"])
                lrs.append(lr_t)
                adjusted_lrs.append(adjusted_lr_t)
            if ps:
                _muon_update(ps, grads, bufs, lrs, adjusted_lrs, group["weight_decay"],
                             group["momentum"], group["nesterov"], tuple(group["ns_coefficients"]),
                             group["ns_steps"], group["eps"])
        if self.vector_optimizer is not None:
            self.vector_optimizer.step()
        return loss

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none)
        if self.vector_optimizer is not None:
            self.vector_optimizer.zero_grad(set_to_none)

    def state_dict(self):
        sd = super().state_dict()
        if self.vector_optimizer is not None:
            sd["vector_optimizer"] = self.vector_optimizer.state_dict()
        return sd

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        vector = state_dict.pop("vector_optimizer", None)
        super().load_state_dict(state_dict)
        for state in self.state.values():  # loading replaces the momentum buffers
            if "momentum_buffer" in state:
                _mark_static(state["momentum_buffer"])
        if vector is not None and self.vector_optimizer is not None:
            self.vector_optimizer.load_state_dict(vector)

def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    Muon uses this to normalize matrix-shaped gradients before applying them.
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return (X.T if transposed else X).to(G.dtype)
