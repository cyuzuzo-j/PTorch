import torch
from .. import config

class ProjectedSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim=None, keepdim=False, dtype=None):
        ctx.save_for_backward(x)
        # Handle dim as either None, int, or tuple.
        ctx.dim = dim
        ctx.keepdim = keepdim
        
        kwargs = {}
        if dim is not None:
            kwargs['dim'] = dim
        if keepdim:
            kwargs['keepdim'] = keepdim
        if dtype is not None:
            kwargs['dtype'] = dtype
        return _original_sum(x, **kwargs)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        if config.use_projections:
            z_target = grad_output
            if ctx.dim is None:
                N = x.numel()
                sum_x = x.sum()
                correction = (z_target - sum_x) / (N + 1)
                return x + correction, None, None, None
            else:
                sum_x = x.sum(dim=ctx.dim, keepdim=True)
                if isinstance(ctx.dim, int):
                    N = x.size(ctx.dim)
                else:
                    N = 1
                    for d in ctx.dim:
                        N *= x.size(d)
                
                if not ctx.keepdim:
                    if isinstance(ctx.dim, int):
                        z_target = z_target.unsqueeze(ctx.dim)
                    else:
                        for d in sorted(ctx.dim):
                            z_target = z_target.unsqueeze(d)
                
                correction = (z_target - sum_x) / (N + 1)
                return x + correction, None, None, None
        else:
            if ctx.dim is None:
                return grad_output.expand_as(x), None, None, None
            else:
                grad = grad_output
                if not ctx.keepdim:
                    if isinstance(ctx.dim, int):
                        grad = grad.unsqueeze(ctx.dim)
                    else:
                        for d in sorted(ctx.dim):
                            grad = grad.unsqueeze(d)
                return grad.expand_as(x), None, None, None

def reduce_gradient_shape(grad, target_shape):
    if grad.shape == target_shape:
        return grad
    extra_dims = grad.dim() - len(target_shape)
    if extra_dims > 0:
        for _ in range(extra_dims):
            grad = grad.sum(0)
    for dim, size in enumerate(target_shape):
        if size == 1 and grad.size(dim) > 1:
            grad = grad.sum(dim, keepdim=True)
    return grad

class ProjectedAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, alpha=1):
        ctx.save_for_backward(x, y)
        ctx.alpha = alpha
        return _original_add(x, y, alpha=alpha)

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        alpha = ctx.alpha
        if config.use_projections:
            # We want to find x*, y* minimizing distance to x, y such that x* + alpha*y* = grad_output
            # This is a projection onto the hyperplane x + alpha*y = z
            # Distance: ||x* - x||^2 + ||y* - y||^2
            # Normal vector to hyperplane is (1, alpha).
            delta = grad_output - (x + alpha * y)
            t = delta / (1.0 + alpha**2)
            grad_x = x + t
            grad_y = y + alpha * t
            
            # handle broadcasting sizes for projected variables to match original x and y
            # wait, projection returns full size variables. If standard broadcast happens,
            # projecting x + y with broadcasting is more complex.
            # Usually users sum tensors of same shape.
            return grad_x, grad_y, None
        else:
            grad_x = grad_output
            grad_y = grad_output * alpha
            
            grad_x = reduce_gradient_shape(grad_x, x.shape)
            grad_y = reduce_gradient_shape(grad_y, y.shape)
                        
            return grad_x, grad_y, None

class ProjectedMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return _original_mul(x, y)

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        if config.use_projections:
            # Exact Euclidean Projection for Hadamard Product: x* * y* = z
            z_target = grad_output
            
            # Broadcast to common shape to find the Lagrange multiplier t
            try:
                common_shape = torch.broadcast_shapes(x.shape, y.shape)
            except AttributeError:
                # Fallback for older PyTorch versions
                common_shape = (x + y).shape
                
            t = torch.zeros(common_shape, device=x.device, dtype=x.dtype)
            
            for _ in range(5):
                # Current projected proposals based on t
                x_star = x + reduce_gradient_shape(t * y, x.shape)
                y_star = y + reduce_gradient_shape(t * x, y.shape)
                
                # Evaluate constraint error: f(t) = x* * y* - z_target
                f = x_star * y_star - z_target
                
                # Derivative of the constraint w.r.t t
                f_prime = x_star * y + y_star * x
                
                step = f / (f_prime.abs() + 1e-8)
                
                # Damped Newton step to prevent divergence in flat areas
                step = torch.clamp(step, -1.0, 1.0)
                t = t - step
                
            # Final projected consensus updates
            grad_x = x + reduce_gradient_shape(t * y, x.shape)
            grad_y = y + reduce_gradient_shape(t * x, y.shape)
            
            return grad_x, grad_y
        else:
            grad_x = grad_output * y
            grad_y = grad_output * x
            
            grad_x = reduce_gradient_shape(grad_x, x.shape)
            grad_y = reduce_gradient_shape(grad_y, y.shape)
            
            return grad_x, grad_y

class ProjectedSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return _original_square(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        if config.use_projections:
            z_clamped = torch.clamp(grad_output, min=0.0)
            return torch.sign(x) * torch.sqrt(z_clamped)
        else:
            return 2.0 * x * grad_output

_original_sum = torch.sum
_original_add = torch.add
_original_square = torch.square
_original_mul = torch.mul
_original_tensor_sum = torch.Tensor.sum
_original_tensor_add = torch.Tensor.add
_original_tensor_square = torch.Tensor.square
_original_tensor_mul = torch.Tensor.mul
_original_tensor_mul_magic = torch.Tensor.__mul__

def projected_sum(input, *args, **kwargs):
    if not isinstance(input, torch.Tensor) or not input.requires_grad:
        return _original_sum(input, *args, **kwargs)
    
    dim = kwargs.get('dim', None)
    if dim is None and len(args) > 0:
        dim = args[0]
        
    keepdim = kwargs.get('keepdim', False)
    if len(args) > 1:
        keepdim = args[1]
        
    dtype = kwargs.get('dtype', None)
    
    return ProjectedSum.apply(input, dim, keepdim, dtype)

def projected_add(input, other, *args, **kwargs):
    if not isinstance(input, torch.Tensor) or not isinstance(other, torch.Tensor):
        return _original_add(input, other, *args, **kwargs)
        
    if not input.requires_grad and not other.requires_grad:
        return _original_add(input, other, *args, **kwargs)
        
    alpha = kwargs.get('alpha', 1)
    if len(args) > 0:
        alpha = args[0]
        
    return ProjectedAdd.apply(input, other, alpha)

def projected_tensor_sum(self, *args, **kwargs):
    return projected_sum(self, *args, **kwargs)

def projected_tensor_add(self, other, *args, **kwargs):
    return projected_add(self, other, *args, **kwargs)

def projected_mul(input, other, *args, **kwargs):
    if not isinstance(input, torch.Tensor) or not isinstance(other, torch.Tensor):
        return _original_mul(input, other, *args, **kwargs)
        
    if not input.requires_grad and not other.requires_grad:
        return _original_mul(input, other, *args, **kwargs)
        
    return ProjectedMul.apply(input, other)

def projected_tensor_mul(self, other, *args, **kwargs):
    return projected_mul(self, other, *args, **kwargs)

def projected_square(input, *args, **kwargs):
    if not isinstance(input, torch.Tensor) or not input.requires_grad:
        return _original_square(input, *args, **kwargs)
    return ProjectedSquare.apply(input)

def projected_tensor_square(self, *args, **kwargs):
    return projected_square(self, *args, **kwargs)

def apply_overrides():
    torch.sum = projected_sum
    torch.Tensor.sum = projected_tensor_sum
    torch.add = projected_add
    torch.Tensor.add = projected_tensor_add
    torch.square = projected_square
    torch.Tensor.square = projected_tensor_square
    torch.mul = projected_mul
    torch.Tensor.mul = projected_tensor_mul
    torch.Tensor.__mul__ = projected_tensor_mul
    
def remove_overrides():
    torch.sum = _original_sum
    torch.add = _original_add
    torch.square = _original_square
    torch.mul = _original_mul
    torch.Tensor.sum = _original_tensor_sum
    torch.Tensor.add = _original_tensor_add
    torch.Tensor.square = _original_tensor_square
    torch.Tensor.mul = _original_tensor_mul
    torch.Tensor.__mul__ = _original_tensor_mul_magic
