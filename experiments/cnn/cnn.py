#########################################
###   reference cnn implementation    ###
#########################################
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))

import pjax
from pjax import nn, optim
from flax import linen

class CNN(linen.Module):
    """Flax CNN model with optional skip connections and max pooling.

    Attributes:
        hidden_features: list of integers specifying channel sizes for each conv layer.
        classes: number of output classses for classification.
        skip: whether to concatenate features from all conv layers.
        max_pool: whether to apply max pooling over spatial dimensions.
        stride: convolution stride for all layers.
    """

    hidden_features: list[int]
    classes: int = 10
    skip: bool = True
    max_pool: bool = True
    stride: int = 1

    @linen.compact
    def __call__(self, x):
        xs = []
        for features in self.hidden_features:
            x = linen.Conv(features, (3, 3), (self.stride, self.stride), "SAME")(x)
            x = linen.relu(x)
            if self.max_pool:
                xs.append(jnp.max(x, axis=(1, 2)))
            else:
                xs.append(jnp.reshape(x, (x.shape[0], -1)))

        if self.skip:
            x = jnp.concatenate(xs, axis=-1)
        else:
            x = xs[-1]

        return linen.Dense(self.classes, kernel_init=linen.initializers.he_normal())(x)


class CNN_pjax(nn.Module):
    """PJAX CNN model with optional skip connections and max pooling.

    Args:
        hidden_features: list of integers specifying channel sizes for conv layers.
        in_features: number of input channels.
        size_2d: spatial size of square input images (height = width).
        classes: number of output classes for classification.
        skip: whether to concatenate features from all conv layers.
        max_pool: whether to apply max pooling over spatial dimensions.
        stride: convolution stride for all layers.
    """

    def __init__(self, hidden_features, in_features, size_2d, classes, skip=True, max_pool=True, stride=1):
        super().__init__()
        self.hidden_features = hidden_features
        self.skip = skip
        self.max_pool = max_pool
        last_f = in_features
        for i, f in enumerate(hidden_features):
            setattr(self, f"conv_{i}", nn.Conv2D(last_f, f, (3, 3), (stride, stride), "SAME"))
            setattr(self, f"relu_{i}", nn.ReLU(f))
            last_f = f

        # calculate output features for the final dense layer
        out_features = 0
        current_h, current_w = size_2d, size_2d
        for f in hidden_features:
            current_h = (current_h + stride - 1) // stride
            current_w = (current_w + stride - 1) // stride
            out_features += f if self.max_pool else current_h * current_w * f
        if not self.skip:
            out_features = hidden_features[-1] if max_pool else current_h * current_w * hidden_features[-1]

        self.out = nn.Linear(out_features, classes)

    def get_params(self, random_key, init_x=None):
        return self.init(random_key)

    def __call__(self, x):
        xs = []
        for i in range(len(self.hidden_features)):
            x = getattr(self, f"conv_{i}")(x)
            x = getattr(self, f"relu_{i}")(x)
            if self.max_pool:
                xs.append(pjax.max(x, axis=(1, 2)))
            else:
                xs.append(pjax.reshape(x, (x.shape[0], -1)))

        if self.skip:
            x = pjax.concatenate(xs, axis=-1)
        else:
            x = xs[-1]

        return self.out(x)
