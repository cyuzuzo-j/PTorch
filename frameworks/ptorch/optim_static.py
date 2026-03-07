import torch                    
                
class AlternatingProjections(torch.optim.Optimizer):
    """
    Minimal projection-based optimizer using alternating projections.
    
    Instead of descending gradients, this exploits native PyTorch `autograd` 
    to pass orthogonal project targets BACKWARDS through the computational graph.
    """
    def __init__(self, params, lr=1.0):
        defaults = dict()
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Updates the parameters using the projected values accumulated in `.grad`.
        
        Requires that `loss.backward(target)` has already been called.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    # Update parameter to the newly projected state
                    # NOTE: p.grad holds the PROJECTION TARGET, not the gradient
                    p.copy_(p.grad)
                    
                    # Clear projection for next iteration
                    p.grad = None

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
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    # p.grad currently holds the projection target p_proj
                    # We want the gradient g to be p - p_proj
                    p.grad.copy_(p.data - p.grad)
        
        # Now apply the standard SGD step using the pseudo-gradients
        return super().step(closure)

class AlternatingProjectionsMomentum(ProjectionSGD):
    """
    Alternating projections optimizer with momentum.
    This is an alias for ProjectionSGD(..., lr=1.0, momentum=0.9).
    """
    def __init__(self, params, lr=1.0, momentum=0.9, **kwargs):
        super().__init__(params, lr=lr, momentum=momentum, **kwargs)

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
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.copy_(p.data - p.grad)
        
        return super().step(closure)

