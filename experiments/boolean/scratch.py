import torch

x = torch.tensor([1.0], requires_grad=True)
w = torch.tensor([2.0], requires_grad=True)

class MyFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        return x * w
        
    @staticmethod
    def backward(ctx, grad_out):
        x, w = ctx.saved_tensors
        print("Backward saw x =", x.item())
        return grad_out * w, grad_out * x

y = MyFunc.apply(x, w)
y.backward(retain_graph=True)

x.data.copy_(torch.tensor([5.0]))
w.data.copy_(torch.tensor([10.0]))

y.backward(retain_graph=True)
