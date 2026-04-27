import re

with open("frameworks/ptorch/core/ops.py", "r") as f:
    content = f.read()

new_class = """
import torch.nn.functional as F

class ConvPatchProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, kernel_size, stride, padding):
        ctx.save_for_backward(input)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        
        # input: (N, C, H, W)
        patches = F.unfold(input, kernel_size, dilation=1, padding=padding, stride=stride)
        
        # patches: (N, C*kH*kW, L)
        # We need output shape: (N, H_out, W_out, C*kH*kW)
        L = patches.shape[-1]
        
        if isinstance(padding, tuple):
            pad_h, pad_w = padding
        else:
            pad_h = pad_w = padding
            
        if isinstance(kernel_size, tuple):
            kH, kW = kernel_size
        else:
            kH = kW = kernel_size
            
        if isinstance(stride, tuple):
            sH, sW = stride
        else:
            sH = sW = stride
            
        H_out = (input.shape[2] + 2 * pad_h - kH) // sH + 1
        W_out = (input.shape[3] + 2 * pad_w - kW) // sW + 1
        
        # Reshape to (N, C*kH*kW, H_out, W_out) then permute to (N, H_out, W_out, C*kH*kW)
        patches = patches.view(input.shape[0], -1, H_out, W_out).permute(0, 2, 3, 1)
        
        return patches

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # grad_output: (N, H_out, W_out, C*kH*kW)
        # Permute back to (N, C*kH*kW, H_out, W_out) and reshape to (N, C*kH*kW, L)
        grad_patches = grad_output.permute(0, 3, 1, 2).reshape(grad_output.shape[0], -1, grad_output.shape[1] * grad_output.shape[2])
        
        # Fold to get sum of targets for each pixel
        grad_input = F.fold(grad_patches, output_size=input.shape[2:], kernel_size=ctx.kernel_size, dilation=1, padding=ctx.padding, stride=ctx.stride)
        
        # Fold ones to get count of overlapping patches for each pixel
        ones = torch.ones_like(grad_patches)
        counts = F.fold(ones, output_size=input.shape[2:], kernel_size=ctx.kernel_size, dilation=1, padding=ctx.padding, stride=ctx.stride)
        
        # Average the targets
        counts = torch.clamp(counts, min=1.0)
        target = grad_input / counts
        
        # Apply projection target propagation logic
        from .ops import process_activation_target
        grad_input = process_activation_target(input, target)
        
        return grad_input, None, None, None

"""

if "class ConvPatchProjection" not in content:
    with open("frameworks/ptorch/core/ops.py", "a") as f:
        f.write(new_class)
    print("Added ConvPatchProjection")
else:
    print("ConvPatchProjection already exists")
