import sys
import os
import torch
import torch.nn as tnn
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))

import pjax
from pjax import nn
import ptorch
import torch.nn as tnn

# Defaults (overridden by config.yaml at runtime)
DEFAULT_CONV_WIDTHS = [4, 8, 8]
WHITEN_KERNEL_SIZE = 2
WHITEN_WIDTH = 2 * 3 * WHITEN_KERNEL_SIZE**2  # 24
SCALING_FACTOR = 1 / 9


# ─── PJAX Model ──────────────────────────────────────────────────────────────

class ConvGroup_PJAX(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = nn.Conv2D(channels_in,  channels_out, kernel_shape=(3, 3), padding="SAME")
        self.pool  = nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
        self.relu1 = nn.ReLU_NB()
        self.conv2 = nn.Conv2D(channels_out, channels_out, kernel_shape=(3, 3), padding="SAME")
        self.relu2 = nn.ReLU_NB()

    def __call__(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x

class CNN_PJAX(nn.Module):
    def __init__(self, classes=10, in_channels=3, conv_widths=None):
        super().__init__()
        widths = conv_widths or DEFAULT_CONV_WIDTHS
        self.num_groups = len(widths)

        self.whiten = nn.Conv2D(in_channels, WHITEN_WIDTH, kernel_shape=(WHITEN_KERNEL_SIZE, WHITEN_KERNEL_SIZE), padding="VALID")
        self.whiten_act = nn.ReLU_NB()

        # Build groups dynamically
        ch_in = WHITEN_WIDTH
        for i, ch_out in enumerate(widths):
            setattr(self, f'group{i+1}', ConvGroup_PJAX(ch_in, ch_out))
            ch_in = ch_out

        self.pool = nn.MaxPool2D(pool_size=(3, 3), strides=(3, 3))

        # Static flatten size: after N groups of pool(2) + final pool(3)
        # spatial dims halve per group then /3.  For both 32×32 and 28×28 → 2×2.
        flatten_size = 2 * 2 * widths[-1]
        self.head = nn.Linear(flatten_size, classes)

    def _forward_features(self, x):
        x = self.whiten(x)
        x = self.whiten_act(x)
        for i in range(self.num_groups):
            x = getattr(self, f'group{i+1}')(x)
        x = self.pool(x)
        return x

    def __call__(self, x):
        x = self._forward_features(x)
        x = pjax.reshape(x, (x.shape[0], -1))
        return self.head(x)


# ─── Pytorch (ptorch) Model ──────────────────────────────────────────────────

class ConvGroup_PTorch(tnn.Module):
    def __init__(self, channels_in, channels_out, alpha=1.0, g=1.0):
        super().__init__()
        self.conv1 = pnn.Conv2D(channels_in,  channels_out, kernel_size=3, padding=1, alpha=alpha, g=g)
        self.pool  = tnn.AvgPool2d(2) # PTorch doesn't have MaxPool2d
        self.relu1 = pnn.ReLU(channels_out)
        self.conv2 = pnn.Conv2D(channels_out, channels_out, kernel_size=3, padding=1, alpha=alpha, g=g)
        self.relu2 = pnn.ReLU(channels_out)

    def forward(self, x):
        x = self.relu1(self.pool(self.conv1(x)))
        x = self.relu2(self.conv2(x))
        return x

class CNN_PTorch(tnn.Module):
    def __init__(self, classes=10, in_channels=3, conv_widths=None, alpha=1.0, g=1.0):
        super().__init__()
        widths = conv_widths or DEFAULT_CONV_WIDTHS
        self.num_groups = len(widths)

        self.whiten = pnn.NativeConv2D(in_channels, WHITEN_WIDTH, kernel_size=WHITEN_KERNEL_SIZE, padding=0, alpha=alpha, g=g)
        self.whiten_relu = pnn.ReLU(WHITEN_WIDTH)

        ch_in = WHITEN_WIDTH
        for i, ch_out in enumerate(widths):
            setattr(self, f'group{i+1}', ConvGroup_PTorch(ch_in, ch_out, alpha=alpha, g=g))
            ch_in = ch_out

        self.pool = tnn.AvgPool2d(3)

        # Predict flattening size via dummy forward
        dummy_in = torch.zeros(1, in_channels, 32 if in_channels == 3 else 28, 32 if in_channels == 3 else 28)
        dummy_out = self._forward_features(dummy_in)
        flatten_size = dummy_out.reshape(1, -1).size(1)

        self.head = pnn.LinearBias(flatten_size, classes, alpha=alpha, g=g)

    def _forward_features(self, x):
        x = self.whiten_relu(self.whiten(x))
        for i in range(self.num_groups):
            x = getattr(self, f'group{i+1}')(x)
        x = self.pool(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.flatten(1)
        return self.head(x) * SCALING_FACTOR


# ─── Torch Baseline Model ────────────────────────────────────────────────────

class ConvGroup_Torch(tnn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = tnn.Conv2d(channels_in,  channels_out, kernel_size=3, padding=1, bias=False)
        self.pool  = tnn.MaxPool2d(2)
        self.bn1   = tnn.BatchNorm2d(channels_out)
        self.relu1 = tnn.ReLU()
        self.conv2 = tnn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1, bias=False)
        self.bn2   = tnn.BatchNorm2d(channels_out)
        self.relu2 = tnn.ReLU()

    def forward(self, x):
        x = self.relu1(self.bn1(self.pool(self.conv1(x))))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x

class CNN_Torch(tnn.Module):
    def __init__(self, classes=10, in_channels=3, conv_widths=None):
        super().__init__()
        widths = conv_widths or DEFAULT_CONV_WIDTHS
        self.num_groups = len(widths)

        self.whiten = tnn.Conv2d(in_channels, WHITEN_WIDTH, kernel_size=WHITEN_KERNEL_SIZE, padding=0, bias=True)
        self.whiten_relu = tnn.ReLU()

        ch_in = WHITEN_WIDTH
        for i, ch_out in enumerate(widths):
            setattr(self, f'group{i+1}', ConvGroup_Torch(ch_in, ch_out))
            ch_in = ch_out

        self.pool = tnn.MaxPool2d(3)

        # Predict flattening size via dummy forward
        dummy_in = torch.zeros(1, in_channels, 32 if in_channels == 3 else 28, 32 if in_channels == 3 else 28)
        dummy_out = self._forward_features(dummy_in)
        flatten_size = dummy_out.reshape(1, -1).size(1)

        self.head = tnn.Linear(flatten_size, classes, bias=False)

    def _forward_features(self, x):
        x = self.whiten_relu(self.whiten(x))
        for i in range(self.num_groups):
            x = getattr(self, f'group{i+1}')(x)
        x = self.pool(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.flatten(1)
        return self.head(x) * SCALING_FACTOR

