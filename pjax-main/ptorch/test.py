import torch

# 1. Define the custom operation and its projection
class SumProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a):
        # The forward pass. We save 'a' for use in the projection later.
        ctx.save_for_backward(a)
        return a.sum()

    @staticmethod
    def backward(ctx, z_target):
        # This acts as your [sum_proj](cci:1://file:///home/cyuzuzo/thesisV5/pjax-main/pjax/core/ops.py:37:0-41:19) function!
        # z_target is the upstream projection coming from parent nodes
        a, = ctx.saved_tensors
        
        # Calculate the projection (same math as pjax sum_proj)
        t = (z_target - a.sum()) / (a.numel() + z_target.numel())
        projected_a = a + t
        
        # Return the projected inputs for the preceding nodes
        return projected_a

# 2. Define a simple Parameter
# requires_grad=True simply tells PyTorch to track it in the graph computation
my_param = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0], requires_grad=True))

# 3. Create a target we want the sum to project onto
target_value = torch.tensor(15.0)

# 4. The Alternating Projections "Optimizer" Update Step
def update_step(param, target):
    # Forward Pass
    output = SumProjection.apply(param)
    
    # Backward Pass (Topological Descent)
    # We pass the target value directly into backward!
    output.backward(target)
    
    # Parameter Update
    with torch.no_grad(): # Don't track this assignment in the graph
        # param.grad now contains the projected state, NOT a gradient
        param.copy_(param.grad)
        
        # Clear the "gradient" (projection) for the next step
        param.grad = None

# 5. Test it
print(f"Initial param: {my_param.data}, sum: {my_param.sum()}")
update_step(my_param, target_value)
print(f"Projected param: {my_param.data}, sum: {my_param.sum()}")
