import torch

class AlternatingProjections(torch.optim.Optimizer):
    """
    Minimal projection-based optimizer using alternating projections.
    
    Instead of descending gradients, this exploits native PyTorch `autograd` 
    to pass orthogonal project targets BACKWARDS through the computational graph.
    """
    def __init__(self, params):
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
