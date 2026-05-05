import sys
import os
import torch
import torch.nn as tnn
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))

import pjax_orr as pjax
from pjax_orr import nn
import ptorch
import ptorch.nn.modules as pnn
import torch.nn as tnn


# ─── PTorch Model ────────────────────────────────────────────────────────────

class SimpleCNN_PTorch(tnn.Module):
    def __init__(self, classes=10, in_channels=3, conv_features=32, alpha=1.0, g=1.0):
        super().__init__()
        
        self.conv1 = pnn.Conv2D(in_channels, conv_features, kernel_size=3, padding=1, alpha=alpha, g=g)
        self.pool1 = pnn.MaxPool2d(2)
        self.relu1 = pnn.LeakyReLU(0.1)

        self.conv2 = pnn.Conv2D(conv_features, conv_features * 2, kernel_size=3, padding=1, alpha=alpha, g=g)
        self.pool2 = pnn.MaxPool2d(2)
        self.relu2 = pnn.LeakyReLU(0.1)

        self.conv3 = pnn.Conv2D(conv_features * 2, conv_features * 4, kernel_size=3, padding=1, alpha=alpha, g=g)
        self.pool3 = pnn.MaxPool2d(2)
        self.relu3 = pnn.LeakyReLU(0.1)

        with torch.no_grad():
            dummy_res = 32 if in_channels == 3 else 28
            dummy_in = torch.zeros(1, in_channels, dummy_res, dummy_res)
            dummy_out = self.relu3(self.pool3(self.conv3(self.relu2(self.pool2(self.conv2(self.relu1(self.pool1(self.conv1(dummy_in)))))))))
            flatten_size = dummy_out.reshape(1, -1).size(1)

        self.head = pnn.Linear(flatten_size, classes, norm="inf")

    def forward(self, x):
        x = self.relu1(self.pool1(self.conv1(x)))
        x = self.relu2(self.pool2(self.conv2(x)))
        x = self.relu3(self.pool3(self.conv3(x)))
        x = x.flatten(1)
        return self.head(x)


# ─── Torch Baseline Model ────────────────────────────────────────────────────

class SimpleCNN_Torch(tnn.Module):
    def __init__(self, classes=10, in_channels=3, conv_features=32):
        super().__init__()
        
        self.conv1 = tnn.Conv2d(in_channels, conv_features, kernel_size=3, padding=1)
        self.pool1 = tnn.MaxPool2d(2)
        self.relu1 = tnn.LeakyReLU(0.1)

        self.conv2 = tnn.Conv2d(conv_features, conv_features * 2, kernel_size=3, padding=1)
        self.pool2 = tnn.MaxPool2d(2)
        self.relu2 = tnn.LeakyReLU(0.1)

        self.conv3 = tnn.Conv2d(conv_features * 2, conv_features * 4, kernel_size=3, padding=1)
        self.pool3 = tnn.MaxPool2d(2)
        self.relu3 = tnn.LeakyReLU(0.1)

        with torch.no_grad():
            dummy_res = 32 if in_channels == 3 else 28
            dummy_in = torch.zeros(1, in_channels, dummy_res, dummy_res)
            dummy_out = self.relu3(self.pool3(self.conv3(self.relu2(self.pool2(self.conv2(self.relu1(self.pool1(self.conv1(dummy_in)))))))))
            flatten_size = dummy_out.reshape(1, -1).size(1)

        self.head = tnn.Linear(flatten_size, classes)

    def forward(self, x):
        x = self.relu1(self.pool1(self.conv1(x)))
        x = self.relu2(self.pool2(self.conv2(x)))
        x = self.relu3(self.pool3(self.conv3(x)))
        x = x.flatten(1)
        return self.head(x)
