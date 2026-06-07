import sys
import os
import torch
import torch.nn as tnn

import ptorch.nn.modules as pnn

from ptorch.nn.modules_experimental import Branch, RMSNorm
from ptorch.config import config as _ptorch_config

class ResidualAdd(torch.autograd.Function):
    """
    Addition for residual connections where the two inputs are independent.
    Properly projects the target onto the addition constraint by splitting
    the residual difference equally between the two branches.
    """
    @staticmethod
    def forward(ctx, skip, out):
        z = skip + out
        ctx.save_for_backward(skip, out, z)
        return z

    @staticmethod
    def backward(ctx, z_target):
        skip, out, z = ctx.saved_tensors
        delta = z_target - z
        
        # Route the full displacement to both addends so the skip path 
        # carries the signal down the backbone with gain 1
        skip_target = skip + delta
        out_target = out + delta
        return skip_target, out_target

class ResidualConvBlock(tnn.Module):
    """ Standard Convolutional Residual Block for PTorch """
    def __init__(self, in_channels, out_channels, alpha=1.0, g=1.0):
        super().__init__()
        # Branch(2) duplicates the tensor so targets propagate correctly backward
        self.branch = Branch(2)
        
        # Main branch
        self.conv = pnn.Conv2D(in_channels, out_channels, kernel_size=3, padding=1, alpha=alpha, g=g)
        self.act = pnn.LeakyReLU(0.1)
        
        # Skip connection branch (requires 1x1 conv if channel dimensions change)
        if in_channels != out_channels:
            self.skip_proj = pnn.Conv2D(in_channels, out_channels, kernel_size=1, padding=0, alpha=alpha, g=g)
        else:
            self.skip_proj = None

    def forward(self, x):
        x_res, x_skip = self.branch(x)
        
        # Main path
        out = self.conv(x_res)
        out = self.act(out)
        
        # Skip path
        if self.skip_proj is not None:
            x_skip = self.skip_proj(x_skip)
            
        # Combine — projection mode uses target-routing semantics; grad mode
        # falls through to plain `+` so autograd produces a true gradient
        # (1,1) into both addends instead of the value-mixed target.
        if _ptorch_config.use_projections:
            return ResidualAdd.apply(x_skip, out)
        return x_skip + out

class ResidualCNN_PTorch(tnn.Module):
    def __init__(self, classes=10, in_channels=3, conv_features=64, alpha=1.0, g=1.0):
        super().__init__()

        # Block 1
        self.block1 = ResidualConvBlock(in_channels, conv_features, alpha=alpha, g=g)
        self.pool1 = pnn.MaxPool2d(2)

        # Block 2
        self.block2 = ResidualConvBlock(conv_features, conv_features * 2, alpha=alpha, g=g)
        self.pool2 = pnn.MaxPool2d(2)

        # Block 3
        self.block3 = ResidualConvBlock(conv_features * 2, conv_features * 4, alpha=alpha, g=g)
        self.pool3 = pnn.MaxPool2d(2)

        with torch.no_grad():
            dummy_res = 32 if in_channels == 3 else 28
            dummy_in = torch.zeros(1, in_channels, dummy_res, dummy_res)
            # Pass through the network to determine the flatten size dynamically
            dummy_out = self.pool3(self.block3(self.pool2(self.block2(self.pool1(self.block1(dummy_in))))))
            flatten_size = dummy_out.reshape(1, -1).size(1)
            print(flatten_size)

        self.head = pnn.Linear(flatten_size, classes, norm="l2")

    def forward(self, x):
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = x.flatten(1)
        return self.head(x)

# ─── PTorch Model ────────────────────────────────────────────────────────────

class SimpleCNN_PTorch(tnn.Module):
    def __init__(self, classes=10, in_channels=3, conv_features=32, alpha=1.0, g=1.0):
        super().__init__()

        self.conv1 = pnn.Conv2D(in_channels, conv_features, kernel_size=3, padding=1, alpha=alpha, g=g)
        self.norm1 = RMSNorm()
        self.pool1 = pnn.MaxPool2d(2)
        self.relu1 = pnn.LeakyReLU(0.1)

        self.conv2 = pnn.Conv2D(conv_features, conv_features * 2, kernel_size=3, padding=1, alpha=alpha, g=g)
        self.norm2 = RMSNorm()
        self.pool2 = pnn.MaxPool2d(2)
        self.relu2 = pnn.LeakyReLU(0.1)

        self.conv3 = pnn.Conv2D(conv_features * 2, conv_features * 4, kernel_size=3, padding=1, alpha=alpha, g=g)
        self.norm3 = RMSNorm()
        self.pool3 = pnn.MaxPool2d(2)
        self.relu3 = pnn.LeakyReLU(0.1)

        with torch.no_grad():
            dummy_res = 32 if in_channels == 3 else 28
            dummy_in = torch.zeros(1, in_channels, dummy_res, dummy_res)
            dummy_out = self._features(dummy_in)
            flatten_size = dummy_out.reshape(1, -1).size(1)

        self.head = pnn.Linear(flatten_size, classes, norm="2")

    def _norm(self, x, norm_layer):
        # RMSNorm acts on the last dim — channel-normalize by permuting NCHW→NHWC,
        # normalizing, then permuting back so spatial structure is preserved.
        x = x.permute(0, 2, 3, 1)
        x = norm_layer(x)
        return x.permute(0, 3, 1, 2)

    def _features(self, x):
        x = self.relu1(self.pool1(self._norm(self.conv1(x), self.norm1)))
        x = self.relu2(self.pool2(self._norm(self.conv2(x), self.norm2)))
        x = self.relu3(self.pool3(self._norm(self.conv3(x), self.norm3)))
        return x

    def forward(self, x):
        x = self._features(x).flatten(1)
        return self.head(x)


# ─── Torch Baseline Model ────────────────────────────────────────────────────

class SimpleCNN_Torch(tnn.Module):
    def __init__(self, classes=10, in_channels=3, conv_features=32):
        super().__init__()

        self.conv1 = tnn.Conv2d(in_channels, conv_features, kernel_size=3, padding=1)
        self.norm1 = tnn.RMSNorm(conv_features)
        self.pool1 = tnn.MaxPool2d(2)
        self.relu1 = tnn.LeakyReLU(0.1)

        self.conv2 = tnn.Conv2d(conv_features, conv_features * 2, kernel_size=3, padding=1)
        self.norm2 = tnn.RMSNorm(conv_features * 2)
        self.pool2 = tnn.MaxPool2d(2)
        self.relu2 = tnn.LeakyReLU(0.1)

        self.conv3 = tnn.Conv2d(conv_features * 2, conv_features * 4, kernel_size=3, padding=1)
        self.norm3 = tnn.RMSNorm(conv_features * 4)
        self.pool3 = tnn.MaxPool2d(2)
        self.relu3 = tnn.LeakyReLU(0.1)

        with torch.no_grad():
            dummy_res = 32 if in_channels == 3 else 28
            dummy_in = torch.zeros(1, in_channels, dummy_res, dummy_res)
            dummy_out = self._features(dummy_in)
            flatten_size = dummy_out.reshape(1, -1).size(1)

        self.head = tnn.Linear(flatten_size, classes)

    def _norm(self, x, norm_layer):
        # RMSNorm acts on the last dim — channel-normalize by permuting NCHW→NHWC,
        # normalizing, then permuting back so spatial structure is preserved.
        x = x.permute(0, 2, 3, 1)
        x = norm_layer(x)
        return x.permute(0, 3, 1, 2)

    def _features(self, x):
        x = self.relu1(self.pool1(self._norm(self.conv1(x), self.norm1)))
        x = self.relu2(self.pool2(self._norm(self.conv2(x), self.norm2)))
        x = self.relu3(self.pool3(self._norm(self.conv3(x), self.norm3)))
        return x

    def forward(self, x):
        x = self._features(x).flatten(1)
        return self.head(x)
