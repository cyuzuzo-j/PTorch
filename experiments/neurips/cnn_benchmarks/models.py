import torch
import torch.nn as tnn

import ptorch.nn.modules as pnn

from ptorch.nn.modules_experimental import Branch, RMSNorm, ResidualAdd
from ptorch.config import config as _ptorch_config

def to_nchw(x, device):
    """(B, H, W, C) numpy batch from the data modules -> (B, C, H, W) tensor."""
    return torch.as_tensor(x, dtype=torch.float32, device=device).permute(0, 3, 1, 2)


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
        self.skip_proj = (pnn.Conv2D(in_channels, out_channels, kernel_size=1, padding=0, alpha=alpha, g=g)
                          if in_channels != out_channels else None)

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
            return ResidualAdd.apply(x_skip, out, "full")
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

        self.head = pnn.Linear(flatten_size, classes, norm="l2")

    def forward(self, x):
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.pool3(self.block3(x))
        x = x.flatten(1)
        return self.head(x)

# ─── ResNet-8 ────────────────────────────────────────────────────────────────
# 8-layer residual CNN with the ResNet-8 topology of He et al. (2016): a 3x3 stem
# conv, three basic blocks of two 3x3 convs (the last two downsample with stride 2
# and a 1x1 conv shortcut) and a linear head. No normalization layers, and the
# final feature map is flattened instead of global-average-pooled: a GAP readout
# moves each of the H*W positions by only 1/(H*W + 1) of the projection
# displacement, which stalls PTorch's convs.

class _BasicBlock(tnn.Module):
    """x -> act(conv3x3 -> act -> conv3x3 (x) + shortcut(x)); shortcut = 1x1 conv
    when the stride or width changes, else the identity."""
    def __init__(self, layers, c_in, c_out, stride):
        super().__init__()
        self.layers = layers
        self.branch = layers["branch"]()
        self.conv1 = layers["conv"](c_in, c_out, 3, stride, 1)
        self.act1 = layers["act"]()
        self.conv2 = layers["conv"](c_out, c_out, 3, 1, 1)
        self.shortcut = layers["conv"](c_in, c_out, 1, stride, 0) if stride != 1 or c_in != c_out else None
        self.act2 = layers["act"]()

    def forward(self, x):
        x_res, x_skip = self.branch(x)
        out = self.conv2(self.act1(self.conv1(x_res)))
        if self.shortcut is not None:
            x_skip = self.shortcut(x_skip)
        return self.act2(self.layers["add"](x_skip, out))


class _ResNet8(tnn.Module):
    def __init__(self, layers, head, classes, in_channels, widths):
        super().__init__()
        self.stem = layers["conv"](in_channels, widths[0], 3, 1, 1)
        self.stem_act = layers["act"]()
        self.blocks = tnn.Sequential(*(
            _BasicBlock(layers, widths[max(i - 1, 0)], c, 1 if i == 0 else 2) for i, c in enumerate(widths)))
        res = 32 if in_channels == 3 else 28
        side = (res + 2 ** (len(widths) - 1) - 1) // 2 ** (len(widths) - 1)  # one stride-2 halving per later block
        self.head = head(widths[-1] * side * side, classes)

    def forward(self, x):
        x = self.blocks(self.stem_act(self.stem(x)))
        return self.head(x.flatten(1))


ACTS = {"leaky": (lambda: pnn.LeakyReLU(0.1), lambda: tnn.LeakyReLU(0.1)), "relu": (pnn.ReLU, tnn.ReLU)}


def _ptorch_add(skip, out):
    # Projection mode routes the full displacement to both addends; gradient
    # mode is a plain sum, so autograd gives the true gradient.
    if _ptorch_config.use_projections:
        return ResidualAdd.apply(skip, out, "full")
    return skip + out


class ResNet8_PTorch(_ResNet8):
    """l2-projected convs and a `head_norm` (l2 | linf) linear head. `branch_mode`
    combines the skip and main-path targets at each block input: "delta_sum"
    (x + sum_i (t_i - x), the backprop fan-out analog) keeps a gain-1 skip; "mean"
    (the consensus projection) halves it per block."""
    def __init__(self, classes=10, in_channels=3, widths=(64, 128, 256), head_norm="l2",
                 branch_mode="delta_sum", act="leaky"):
        layers = {
            "conv": lambda i, o, k, s, p: pnn.Conv2D(i, o, kernel_size=k, stride=s, padding=p),
            "act": ACTS[act][0],
            "branch": lambda: Branch(2, mode=branch_mode),
            "add": _ptorch_add,
        }
        super().__init__(layers, lambda i, o: pnn.Linear(i, o, norm=head_norm), classes, in_channels, widths)


class ResNet8_Torch(_ResNet8):
    def __init__(self, classes=10, in_channels=3, widths=(64, 128, 256), act="leaky"):
        layers = {
            "conv": lambda i, o, k, s, p: tnn.Conv2d(i, o, kernel_size=k, stride=s, padding=p),
            "act": ACTS[act][1],
            "branch": lambda: (lambda x: (x, x)),
            "add": lambda skip, out: skip + out,
        }
        super().__init__(layers, tnn.Linear, classes, in_channels, widths)
