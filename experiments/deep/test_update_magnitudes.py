import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))

import torch
import torch.nn as tnn
import matplotlib.pyplot as plt

from ptorch.nn.modules import Linear, CrossEntropyLoss,LinearOrth, LinearHybrid
from ptorch.optim_static import AlternatingProjections, ProjectionMuon
from experiments.shared.data import MNISTDataModule
import ptorch.config as ptorch_config
# ─── Configuration ────────────────────────────────────────────────────────────
RANDOM_SEED  = 42
BATCH_SIZE   = 256
MAX_STEPS    = 100
HIDDEN       = [256,256,256,256]

# Tuple format: (display_name, optimizer_class, use_grad, dtp, norm, batch_size)
OPTIMIZERS = [
    ("AP dtp=F bs=16 l2", ProjectionMuon, True, False, "l2", 16),
    ("AP dtp=F bs=16 linf", ProjectionMuon, True, False, "linf", 16),
    ("AP dtp=T bs=16 l2", ProjectionMuon, True, True, "l2", 16),
    ("AP dtp=T bs=16 linf", ProjectionMuon, True, True, "linf", 16),
    ("AP dtp=F bs=2 l2", ProjectionMuon, True, False, "l2", 2),
    ("AP dtp=F bs=2 linf", ProjectionMuon, True, False, "linf", 2),
    ("AP dtp=T bs=2 l2", ProjectionMuon, True, True, "l2", 2),
    ("AP dtp=T bs=2 linf", ProjectionMuon, True, True, "linf", 2),
    ("MUON GRAD dtp=T bs=256 l2", ProjectionMuon, False, True, "l2", 256),
]



# ─── Model ────────────────────────────────────────────────────────────────────
class MLP(tnn.Module):
    def __init__(self, hidden_features, in_features, classes, dtp=True, norm="l2", batch_size=256):
        super().__init__()
        self.dtp = dtp
        self.batch_size = batch_size
        self.layers = tnn.ModuleList()
        self.layers.append(Linear(in_features, hidden_features[0]))
        last_f = hidden_features[0]
        for f in hidden_features[1:]:
            self.layers.append(Linear(last_f, f, norm=norm, dtp=dtp))
            last_f = f
        self.out = Linear(last_f, classes, norm=norm)

        
        self.captured_aproj = {} # Will store layer_idx -> norm
        self._setup_hooks()

    def _setup_hooks(self):
        """Attaches hooks to capture ||A_proj - A_old|| during fwd/bwd passes."""
        linear_layers = [self.layers[i] for i in range(len(self.layers))] + [self.out]
        
        def make_fwd_hook(idx):
            def hook(module, inp, out):
                a_old = out.detach().clone()
                def grad_hook(grad):
                    # grad is the projected activation (A_proj) flowing backward
                    if ptorch_config.config.use_projections:
                        self.captured_aproj[idx] = (grad - a_old).norm().item() / self.batch_size
                    else:
                        self.captured_aproj[idx] = grad.norm().item() / self.batch_size
                if out.requires_grad:
                    out.register_hook(grad_hook)
            return hook

        for i, layer in enumerate(linear_layers):
            layer.register_forward_hook(make_fwd_hook(i))

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        for layer in self.layers:
            x = layer(x)
        return self.out(x)

# ─── Main Execution ───────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(RANDOM_SEED)

    dataset = MNISTDataModule(batch_size=BATCH_SIZE, seed=RANDOM_SEED)
    train_iter = dataset.train_iterator()
    
    # 1. Initialize models and optimizers upfront
    active_runs = []
    for opt_name, opt_class, use_grad, dtp, norm, run_bs in OPTIMIZERS:
        model = MLP(HIDDEN, 28 * 28, 10, dtp=dtp, norm=norm, batch_size=run_bs).to(device)
        optimizer = opt_class(model.parameters(), )
        active_runs.append((opt_name, model, optimizer, use_grad, run_bs))


    # 2. Setup interactive plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 5))
    num_layers = len(HIDDEN) + 1
    layer_indices = list(range(num_layers))
    loss = CrossEntropyLoss()
    # 3. Training Loop
    for step in range(MAX_STEPS):
        x, y = next(train_iter)
        x = torch.tensor(x, dtype=torch.float32, device=device)
        y = torch.tensor(y, dtype=torch.long, device=device)

        ax.clear()

        for opt_name, model, optimizer, use_grad, run_bs in active_runs:
            # Slice data for the specific batch size
            x_run = x[:run_bs]
            y_run = y[:run_bs]
            with ptorch_config.config.projections(use_grad):

                model.train()
                optimizer.zero_grad()
                
                # Forward
                logits = model(x_run)
                y_one_hot = tnn.functional.one_hot(y_run, num_classes=logits.shape[-1]).float()
                
                # Projection / Backward
                projected = loss(logits, y_one_hot)
                projected.sum().backward()
                optimizer.step()

                # Extract norms sorted by layer index
                norms = [model.captured_aproj.get(i, 0.0) for i in layer_indices]
                
                # Plot this optimizer's line for the current step
                ax.plot(layer_indices, norms, marker='o', label=opt_name)

        # Formatting the plot
        ax.set_xticks(layer_indices)
        ax.set_xlabel("Layer Number (0 = Input-side, Max = Output)")
        ax.set_ylabel("|| A_proj - A_old ||₂ / batch_size")
        ax.set_yscale('log')
        ax.set_title(f"Hidden State Projection Norms - Step {step + 1}/{MAX_STEPS}")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

        # Update the plot live
        plt.draw()
        plt.pause(10) 

    # Keep the plot open at the very end
    plt.show()

if __name__ == "__main__":
    main()