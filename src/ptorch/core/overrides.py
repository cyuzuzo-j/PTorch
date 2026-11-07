import torch
import torch.nn.functional as F
from torch.utils._pytree import tree_leaves, tree_map
from ..config import config
from .ops import ParameterOperand, defer_conversion, is_parameter_operand, parameter_operand

# Projection-mode targets are joint graph projections (the output target counts as
# one more unit-weight coordinate, as in ops.py), so x+y, x*y, x.square() and
# stack([x, y]).sum(0) agree.  Constant operands (needs_input_grad False) stay fixed.
# Autograd passes these targets between nodes; a Parameter operand, or a view or copy of
# one, goes in through parameter_operand, so that its .grad holds the residual like every
# parameter's. Here that residual is p - target, a subtraction: these ops return targets.
#
# Covered: the ops of _PATCHES, sub and the in-place ops of _PARAMETER_PATCHES, the copying
# ops of _COPY_PATCHES and _SELF_COPY_PATCHES, and ProjectionModule inputs (nn/modules.py).
# Not covered, with the raw target reaching .grad and no error: F.embedding (see below),
# value-changing native ops on a Parameter (p.exp(), p @ x, p / 2), writes of a Parameter
# into a tensor (x[i] = p, copy_, index_add, index_put, scatter, masked_scatter), backward
# started on a view, and code bound to the native op before ptorch was imported
# (from torch import cat). A partial selection of an op that mixed a Parameter with an
# activation gives p - 0 on what it leaves out, as such an output counts as an activation.
# Operator syntax (x + p, p[i]) inside torch.compile bypasses every patch here.

def _consensus(v, ref):  # mean of a broadcast operand's per-output targets
    return v.sum_to_size(ref.shape) * (ref.numel() / v.numel())

def _fit(a, b, z):  # exact argmin_a* |a*-a|^2 + |a* b - z|^2, a* broadcast over z
    return (a + (b * z).sum_to_size(a.shape)) / (1 + (b * b).expand_as(z).sum_to_size(a.shape))

class ProjectedSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, args, kwargs):
        ctx.save_for_backward(x)
        dim = kwargs.get('dim', kwargs.get('axis', args[0] if args else None))
        dims = {d % x.dim() for d in ([dim] if isinstance(dim, int) else dim or ())} or set(range(x.dim()))
        ctx.shape = [1 if i in dims else s for i, s in enumerate(x.shape)]
        return _ORIG[torch, 'sum'](x, *args, **kwargs)

    @staticmethod
    def backward(ctx, z):
        x, = ctx.saved_tensors
        s = x.sum(tuple(i for i, n in enumerate(ctx.shape) if n == 1), keepdim=True)
        return x + (z.reshape(ctx.shape) - s) / (x.numel() // s.numel() + 1), None, None

class ProjectedAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, alpha):
        ctx.save_for_backward(x, y)
        ctx.alpha = alpha
        return _ORIG[torch, 'add'](x, y, alpha=alpha)

    @staticmethod
    def backward(ctx, z):
        # A broadcast operand of multiplicity m has inverse stiffness m; this is the exact
        # joint projection when the residual is uniform over each broadcast group.
        x, y = ctx.saved_tensors
        a, (wx, wy) = ctx.alpha, [g * z.numel() // t.numel() for g, t in zip(ctx.needs_input_grad, (x, y))]
        lam = (z - (x + a * y)) / (1 + wx + a * a * wy)
        return (_consensus(x + wx * lam, x) if wx else None,
                _consensus(y + a * wy * lam, y) if wy else None, None)

class ProjectedMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return _ORIG[torch, 'mul'](x, y)

    @staticmethod
    def backward(ctx, z):
        # min |x*-x|^2 + |y*-y|^2 + |x*y* - z|^2; with one factor fixed it is _fit.
        x, y = ctx.saved_tensors
        nx, ny = ctx.needs_input_grad
        if not (nx and ny):
            return _fit(x, y, z) if nx else None, _fit(y, x, z) if ny else None
        # KKT: x* = x + t y*, y* = y + t x*, t = z - x*y*.  Rotated, x*y* = (u^2 - v^2)/2 with
        # u = U p, v = V p/(2p-1); flip so the root has p >= 1, where Newton in p is safe.
        r = 2 ** 0.5
        U, V, pos = (x + y) / r, (x - y) / r, z >= x * y
        A, B, Z = torch.where(pos, U, V), torch.where(pos, V, U), torch.where(pos, z, -z)
        p = torch.ones_like(Z)
        for _ in range(10):
            q = p / (2 * p - 1)
            p = p - ((A * A * p * p - B * B * q * q) / 2 + 1 - 1 / p - Z) / (A * A * p + (B * B * q ** 3 + 1) / (p * p))
        a, b = A * p, B * p / (2 * p - 1)
        U, V = torch.where(pos, a, b), torch.where(pos, b, a)
        xs, ys = _consensus((U + V) / r, x), _consensus((U - V) / r, y)
        for _ in range(0 if x.shape == y.shape else 3):  # broadcast: refine by exact block solves
            xs = _fit(x, ys, z)
            ys = _fit(y, xs, z)
        return xs, ys

class ProjectedSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _ORIG[torch, 'square'](x)

    @staticmethod
    def backward(ctx, z):
        # argmin_u (u-x)^2 + (u^2-z)^2: the largest root of 2u^3 + (1-2z)u - |x|, signed
        # like x.  Newton from an upper bound decreases monotonically (convex for u > 0).
        x, = ctx.saved_tensors
        ax = x.abs()
        u = torch.maximum(ax, z.clamp(min=0).sqrt())
        for _ in range(8):
            u = u - (2 * u ** 3 + (1 - 2 * z) * u - ax) / (6 * u * u + 1 - 2 * z)
        return torch.where(x < 0, -u, u)

def _t(v, ref):  # a Python scalar becomes a 0-d tensor on ref's device (same type promotion)
    return v if isinstance(v, torch.Tensor) else torch.full((), v, device=ref.device)

def _sum(x, *args, **kwargs):
    return ProjectedSum.apply(x, args, kwargs)

def _add(x, y, alpha=1):
    return ProjectedAdd.apply(_t(x, y), _t(y, x), alpha)

def _mul(x, y):
    return ProjectedMul.apply(_t(x, y), _t(y, x))

_PATCHES = [(torch, 'sum', _sum), (torch.Tensor, 'sum', _sum), (torch, 'square', ProjectedSquare.apply),
            (torch.Tensor, 'square', ProjectedSquare.apply), (torch, 'add', _add), (torch.Tensor, 'add', _add),
            (torch.Tensor, '__add__', _add), (torch.Tensor, '__radd__', _add),
            (torch.Tensor, '__sub__', lambda x, y: _add(x, y, alpha=-1)),
            (torch.Tensor, '__rsub__', lambda x, y: _add(y, x, alpha=-1)), (torch, 'mul', _mul),
            (torch.Tensor, 'mul', _mul), (torch.Tensor, '__mul__', _mul), (torch.Tensor, '__rmul__', _mul)]

def _sub(x, y, alpha=1):
    return _add(x, y, alpha=-alpha)

def _inplace(proj):  # x's old value is saved by proj, so give it a copy
    return lambda x, *args, **kwargs: x.copy_(proj(x.clone(), *args, **kwargs))

# These stay native on activations; a Parameter operand must not bypass the conversion.
# The in-place ones only look at the operand: x is overwritten, so it must not be copied.
_PARAMETER_PATCHES = [(torch, 'sub', _sub, 0), (torch.Tensor, 'sub', _sub, 0)] + [
    (torch.Tensor, name, _inplace(proj), 1)
    for names, proj in ((('add_', '__iadd__'), _add), (('sub_', '__isub__'), _sub), (('mul_', '__imul__'), _mul))
    for name in names]

# Ops whose output coordinates are coordinates of their tensor operands (or constants):
# they copy, select or rearrange and compute nothing. On a Parameter they stay native
# with their output converted, see _copy_of_parameter. Views need no patch: they keep
# _base and are converted by the next op here. F.embedding stays out on purpose:
# nn.Embedding + Conversion passes a true gradient, which a conversion would undo.
# A new op belongs here only if it meets that first sentence.
_SELECTIONS = ('clone reshape flatten ravel tile repeat_interleave flip fliplr flipud roll rot90 index_select '
               'gather take take_along_dim masked_select tril triu').split()
_COPY_PATCHES = [(torch, n) for n in _SELECTIONS + 'where masked_fill cat concat concatenate stack hstack vstack '
                 'dstack column_stack'.split()] + [(torch.Tensor, 'where'), (torch.Tensor, 'masked_fill'), (F, 'pad')]
# Tensor methods whose only differentiable operand is self: a cheaper check, as these are hot.
_SELF_COPY_PATCHES = [(torch.Tensor, n) for n in _SELECTIONS + (
    '__getitem__ to type type_as double float half bfloat16 cpu cuda contiguous repeat reshape_as').split()]
_ORIG = {(o, n): getattr(o, n) for o, n, *_ in _PATCHES + _PARAMETER_PATCHES + _COPY_PATCHES + _SELF_COPY_PATCHES}

def _dispatch(orig, proj):
    # Gradient mode (and no-grad code, including our own backwards) runs the native op.
    def op(*args, **kwargs):
        if config.use_projections and torch.is_grad_enabled() and any(
                isinstance(a, torch.Tensor) and a.requires_grad for a in args[:2]):
            return proj(*map(parameter_operand, args), **kwargs)
        return orig(*args, **kwargs)
    return op

def _dispatch_parameter(orig, proj, first):
    def op(*args, **kwargs):
        if config.use_projections and torch.is_grad_enabled() and any(map(is_parameter_operand, args[:2])):
            return proj(*args[:first], *map(parameter_operand, args[first:]), **kwargs)
        return orig(*args, **kwargs)
    return op

def _has_parameter(args, kwargs):
    for a in (*args, *kwargs.values()):
        if any(map(is_parameter_operand, a)) if isinstance(a, (list, tuple)) else is_parameter_operand(a):
            return True
    return False

def _copy_of_parameter(orig, args, kwargs):
    # A target z on output y reaches each Parameter as the residual J^T (y - z): the
    # conversion sits on the output, because J^T scatters zeros, so converting the operand
    # would give p - 0 on the coordinates that were not selected. Other operands that need
    # a grad are activations and get J^T z, from a second run of the op without Parameters.
    def run(differentiable):
        a, k = tree_map(lambda t: t.detach() if isinstance(t, torch.Tensor) and t.requires_grad
                        and not differentiable(t) else t, (args, kwargs))
        return orig(*a, **k)
    y = run(is_parameter_operand)
    if not (isinstance(y, torch.Tensor) and y.requires_grad):  # e.g. p.long(), x.to(p)
        return orig(*args, **kwargs)
    if is_parameter_operand(y):  # p itself or a view of it: the next op sees that
        return y
    operands = [a for a in tree_leaves((args, kwargs)) if isinstance(a, torch.Tensor) and a.requires_grad]
    mixed = not all(map(is_parameter_operand, operands))
    z = run(lambda a: not is_parameter_operand(a)) if mixed else None
    for a in operands:
        defer_conversion(a)
    return ParameterOperand.apply(y, z)

def _dispatch_copy(orig):
    def op(*args, **kwargs):
        if torch.is_grad_enabled() and config.use_projections and _has_parameter(args, kwargs):
            return _copy_of_parameter(orig, args, kwargs)
        return orig(*args, **kwargs)
    return op

def _dispatch_self_copy(orig):
    def op(x, *args, **kwargs):
        if x.requires_grad and torch.is_grad_enabled() and config.use_projections and is_parameter_operand(x):
            return _copy_of_parameter(orig, (x, *args), kwargs)
        return orig(x, *args, **kwargs)
    return op

def apply_overrides():
    for o, n, proj in _PATCHES:
        setattr(o, n, _dispatch(_ORIG[o, n], proj))
    for o, n, proj, first in _PARAMETER_PATCHES:
        setattr(o, n, _dispatch_parameter(_ORIG[o, n], proj, first))
    for o, n in _COPY_PATCHES:
        setattr(o, n, _dispatch_copy(_ORIG[o, n]))
    for o, n in _SELF_COPY_PATCHES:
        setattr(o, n, _dispatch_self_copy(_ORIG[o, n]))

def remove_overrides():
    for (o, n), orig in _ORIG.items():
        setattr(o, n, orig)
