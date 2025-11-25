import jax
import pjax
from pjax import nn, optim
from data import (
    MNISTDataModule,
) 

dataset = MNISTDataModule(batch_size=32)
train_data =dataset.train_dataloader()

# 1. Define the model
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

# 2. Initialize model and optimizer
key = jax.random.key(0)
model = CNN_pjax([32], in_features=1, size_2d=28, classes=10, max_pool=True, stride=1)
params = model.init(key)

optimizer = optim.AlternatingProjections(steps_per_update=50)
# 3. Define a training step
@jax.jit
def train_step(params, x, y):
    def apply_fn(params):
        logits = model.apply(params, x)
        y_one_hot = jax.nn.one_hot(y, num_classes=10)
        return pjax.cross_entropy(logits, y_one_hot)

    updated_params, loss = optimizer.update(apply_fn, params)
    return updated_params, loss



# 4. Training loop
losses  = []
losses2 = []
losses3 = []
for step in range(1):
    for i, (x,y)  in enumerate(train_data):
        print("Batch", i)
        params, loss = train_step(params, x, y)
        losses.append(loss)
        if i > 10:
            break
    print("Step:", step, "Loss:", loss)
        
import matplotlib.pyplot as plt
plt.plot(losses)
plt.show()