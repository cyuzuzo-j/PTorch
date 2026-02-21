#########################################
###   reference mlp implementation    ###
#########################################
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pjax
from pjax import nn, optim
from flax import linen


# --- Model Definitions ---
class MLP(linen.Module):
    """Flax MLP model with optional skip connections.

    Attributes:
        hidden_features: list of integers specifying the number of units in each hidden layer.
        classes: number of output classes for classification.
        skip: whether to use skip connections by concatenating all hidden layer outputs.
    """

    hidden_features: list[int]
    classes: int = 10
    skip: bool = True

    @linen.compact
    def __call__(self, x):
        # Flatten input for MLP
        x = x.reshape((x.shape[0], -1))
        xs = []
        for features in self.hidden_features:
            x = linen.Dense(features, kernel_init=linen.initializers.he_normal())(x)
            x = linen.relu(x)
            xs.append(x)

        if self.skip:
            x = jnp.concatenate(xs, axis=-1)

        return linen.Dense(self.classes, kernel_init=linen.initializers.he_normal())(x)

    def get_params(random_key):
        return None


class MLP_pjax(nn.Module):
    """PJAX MLP model with optional skip connections.

    Args:
        hidden_features: list of integers specifying hidden layer sizes.
        in_features: number of input features (flattened input size).
        classes: number of output classes for classification.
        skip: whether to concatenate all hidden layer outputs before final layer.
    """


    def __init__(self, hidden_features, in_features, classes, skip=True):
        super().__init__()
        self.hidden_features = hidden_features
        self.skip = skip
        last_f = in_features
        for i, f in enumerate(hidden_features):
            setattr(self, f"dense_{i}", nn.LinearOld(last_f, f))
            setattr(self, f"relu_{i}", nn.ReLU(f))
            last_f = f

        out_features = sum(hidden_features) if skip else hidden_features[-1]
        self.out = nn.LinearOld(out_features, classes)
    
    def get_params(self, random_key, init_x=None):
        return self.init(random_key)

    def __call__(self, x):
        # Flatten input for MLP
        x = pjax.reshape(x, (x.shape[0], -1))
        xs = []
        for i in range(len(self.hidden_features)):
            x = getattr(self, f"dense_{i}")(x)
            x = getattr(self, f"relu_{i}")(x)
            xs.append(x)

        if self.skip:
            x = pjax.concatenate(xs, axis=-1)

        return self.out(x)
