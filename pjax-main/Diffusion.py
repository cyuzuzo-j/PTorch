import jax
import jax.numpy as jnp
import pjax
from pjax import nn, optim
from data import MNISTDataModule
import matplotlib.pyplot as plt

def get_mnist(batch_size=128):
    dataset = MNISTDataModule(batch_size=batch_size)
    return dataset.train_dataloader()

T = 100  # fewer steps for speed
betas = jnp.linspace(1e-4, 0.01, T)
alphas = 1.0 - betas
alpha_bars = jnp.cumprod(alphas)

class TinyMLP(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(28*28, 28*28)
        self.t_dense1 = nn.Linear(1, hidden_dim)
        self.t_relu1 = nn.ReLU(hidden_dim)
        self.t_dense2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.h_dense1 = nn.Linear(28*28 + hidden_dim, hidden_dim)
        self.h_relu1 = nn.ReLU(hidden_dim)
        self.h_dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.h_relu2 = nn.ReLU(hidden_dim)
        self.out_dense = nn.Linear(hidden_dim, 28*28)

    def __call__(self, x, t):
        # Flatten image
        x_flat =  self.fc(x)
        # Embed timestep
        t_emb = self.t_dense1(t)
        t_emb = self.t_relu1(t_emb)
        t_emb = self.t_dense2(t_emb)
        # Concatenate
        
        h = pjax.concatenate([x_flat, t_emb], axis=-1)
        h = self.h_dense1(h)
        h = self.h_relu1(h)
        h = self.h_dense2(h)
        h = self.h_relu2(h)
        out = self.out_dense(h)
        return out

def q_sample(key, x0, t):
    noise = jax.random.normal(key, x0.shape)
    sqrt_ab = jnp.sqrt(alpha_bars[t])[:, None]
    sqrt_omb = jnp.sqrt(1 - alpha_bars[t])[:, None]
    return sqrt_ab * x0 + sqrt_omb * noise, noise

def train_step(params, key, batch, model, optimizer):
    def loss_fn(params):
        bs = batch.shape[0]
        key_t, key_n = jax.random.split(key)
        t = jax.random.randint(key_t, (bs,), 0, T)
        x_t, noise = q_sample(key_n, batch, t)
        AAt = t.astype(jnp.float32).reshape(-1, 1)
        pred_noise = model.apply(params, x_t, AAt)
        return pjax.means_squared_error(pred_noise, noise)  
    
    updated_params, loss = optimizer.update(loss_fn, params)
    return updated_params, loss

# Initialize model and optimizer
rng = jax.random.PRNGKey(0)
model = TinyMLP(hidden_dim=128)
params = model.init(rng)
optimizer = optim.DouglasRachfordMonumentum(steps_per_update=50)

dataset = get_mnist(batch_size=128)

# Train without enumerate to avoid tracer issues
train_step_jit = jax.jit(lambda params,key, batch: train_step(params, key, batch, model, optimizer))
for epoch in range(10):
    batch_iter = iter(dataset)
    for i in range(11):  # fixed number of iterations
        try:
            batch, _ = next(batch_iter)
            rng, key = jax.random.split(rng)
            batch = batch.reshape(batch.shape[0], 28*28)
            
            params, loss = train_step_jit(params, key, batch)
            if i == 3:
                print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss:.4f}")
                break
        except StopIteration:
            break
    print(f"Epoch {epoch+1} done.")

def p_sample(key, model, params, x, t_scalar):
    """t_scalar is a Python int, not a JAX array"""
    t_array = jnp.full((x.shape[0],), t_scalar, dtype=jnp.float32)
    t_array = t_array.astype(jnp.float32).reshape(-1, 1)
    pred_noise = model.apply(params, x, t_array)
    alpha_t = alphas[t_scalar]
    alpha_bar_t = alpha_bars[t_scalar]
    beta_t = betas[t_scalar]
    mean = (1 / jnp.sqrt(alpha_t)) * (x - (beta_t / jnp.sqrt(1 - alpha_bar_t)) * pred_noise)
    
    # Use jax.lax.cond instead of Python if
    noise = jax.random.normal(key, x.shape)
    return jax.lax.cond(
        t_scalar > 0,
        lambda _: mean + jnp.sqrt(beta_t) * noise,
        lambda _: mean,
        None
    )

def sample_images(rng, model, params, n=16):
    x = jax.random.normal(rng, (n, 28* 28))
    
    # Use Python loop since we're calling with concrete t values
    for t in reversed(range(T)):
        rng, key = jax.random.split(rng)
        x = p_sample(key, model, params, x, t)
    
    return jnp.clip(x, 0, 1)

samples = sample_images(rng, model, params, n=16)

plt.figure(figsize=(6,6))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(samples[i].reshape(28,28), cmap='gray')
    plt.axis('off')
plt.show()
