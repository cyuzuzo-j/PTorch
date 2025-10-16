import jax
import pjax
from pjax import nn, optim
from data import (
    MNISTDataModule,
) 

dataset = MNISTDataModule(batch_size=128)
train_data =dataset.train_dataloader()
# 1. Define the model
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes):
        super().__init__()
        self.dense1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU(hidden_features)
        self.dense2 = nn.Linear(hidden_features, num_classes)

    def __call__(self, x):
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        return x

# 2. Initialize model and optimizer
key = jax.random.key(0)
model = MLP(in_features=784, hidden_features=256, num_classes=10)
model2 = MLP(in_features=784, hidden_features=256, num_classes=10)
model3 = MLP(in_features=784, hidden_features=256, num_classes=10)
params = model.init(key)
params2 = model2.init(key)
params3 = model3.init(key)

optimizer = optim.AlternatingProjections(steps_per_update=50)
optimizer2 = optim.AlternatingProjectionsMonumentum(steps_per_update=50)
optimizer3 = optim.DouglasRachford(steps_per_update=50)
# 3. Define a training step
@jax.jit
def train_step(params, x, y):
    def apply_fn(params):
        logits = model.apply(params, x)
        y_one_hot = jax.nn.one_hot(y, num_classes=10)
        return pjax.cross_entropy(logits, y_one_hot)

    updated_params, loss = optimizer.update(apply_fn, params)
    return updated_params, loss

@jax.jit
def train_step2(params, x, y):
    def apply_fn(params):
        logits = model.apply(params, x)
        y_one_hot = jax.nn.one_hot(y, num_classes=10)
        return pjax.cross_entropy(logits, y_one_hot)

    updated_params, loss = optimizer2.update(apply_fn, params)
    return updated_params, loss

@jax.jit
def train_step3(params, x, y):
    def apply_fn(params):
        logits = model.apply(params, x)
        y_one_hot = jax.nn.one_hot(y, num_classes=10)
        return pjax.cross_entropy(logits, y_one_hot)

    updated_params, loss = optimizer3.update(apply_fn, params)
    return updated_params, loss

# 4. Training loop
losses  = []
losses2 = []
losses3 = []
for step in range(1):
    for i, (x,y)  in enumerate(train_data):
        x = x.reshape(x.shape[0], -1)  # flatten the images
        print("aaa")
        params, loss = train_step(params, x, y)
        params2, loss2 = train_step2(params2, x, y)
        params3, loss3 = train_step3(params3, x, y)
        losses.append(loss)
        losses2.append(loss2)
        losses3.append(loss3)
        if i == 4:
            break
        print("batch step", i)
    print("Step:", step, "Loss:", loss)
        
import matplotlib.pyplot as plt
plt.plot(losses)
plt.plot(losses2, color='red')
plt.plot(losses3, color='green')
plt.legend(['AP', 'AP-Monumentum', 'AP-Adam'])
plt.show()