import jax
import pjax
from pjax import nn, optim
from data import (
    MNISTDataModule,
) 

dataset = MNISTDataModule(batch_size=5)
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
            setattr(self, f"conv_{i}", nn.FftConv2D(28,28,1,1, 3))
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

optimizer = optim.AlternatingProjections(steps_per_update=1)
# 3. Define a training step
@jax.jit
def train_step(params, x, y):
    def apply_fn(params):
        logits = model.apply(params, x)
        y_one_hot = jax.nn.one_hot(y, num_classes=10)
        y_one_hot = y_one_hot.astype(jax.numpy.complex64)
        #jax.debug.print("Logits: {}", logits.value)
        return pjax.cross_entropy(logits, y_one_hot)

    updated_params, loss = optimizer.update(apply_fn, params)
    return updated_params, loss



# 4. Training loop
losses  = []
losses2 = []
losses3 = []
for step in range(10):
    avg_loss = 0
    for i, (x,y)  in enumerate(train_data):
        params, loss = train_step(params, x, y)
        print("Loss:", loss)
        avg_loss += loss
        if i>150:
            break
    avg_loss /= (i+1)
    print("Step:", step, "Loss:", loss)
        
import matplotlib.pyplot as plt
plt.plot(losses)

# Prediction
print("\n--- Prediction on Test Data ---")
# Get one batch
x_test, y_test = next(iter(train_data))

# Run model
logits = model.apply(params, x_test)

# Check if logits are complex
if jax.numpy.iscomplexobj(logits):
    print("Logits are complex. Using real part for prediction.")
    predictions = jax.numpy.argmax(logits.real, axis=-1)
else:
    predictions = jax.numpy.argmax(logits, axis=-1)

print("Predictions:", predictions)
print("Actual labels:", y_test)

# Calculate accuracy on this batch
accuracy = jax.numpy.mean(predictions == y_test)
print(f"Batch Accuracy: {accuracy:.2f}")

# Visualize
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i in range(min(5, len(x_test))):
    img = x_test[i]
    if img.shape[-1] == 1:
        img = img.reshape(28, 28)
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"Pred: {predictions[i]}, True: {y_test[i]}")
    axes[i].axis('off')
plt.show()
