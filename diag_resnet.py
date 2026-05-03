"""Quick diagnostic for bench_ptorch_resnet: check targets, gradients, weight updates."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'frameworks')))

import torch
import torch.nn.functional as F
import ptorch.optim_static as ptorch_optim_static
import ptorch.nn.modules as pnn
import ptorch.config as ptorch_config
from ptorch.nn.modules import HardMarginLoss
from ptorch.core.overrides import apply_overrides

apply_overrides()
ptorch_config.use_projections = True

from experiments.cnn_benchmarks.bench_ptorch_resnet import ResNet8_PTorch, ProjectedGlobalAvgPool
from experiments.shared.data import InfiniteCifarDataModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

torch.manual_seed(42)
model = ResNet8_PTorch(classes=10, in_channels=3).to(device)

ds = InfiniteCifarDataModule(batch_size=64, data_dir="./dataset", seed=42)
train_iter = ds.train_iterator()

optimizer = ptorch_optim_static.ProjectionMuon(model.parameters(), lr=0.001)

for step in range(20):
    x_np, y_np = next(train_iter)
    x_batch = torch.tensor(x_np, dtype=torch.float32, device=device)
    if x_batch.shape[-1] in [1, 3]:
        x_batch = x_batch.permute(0, 3, 1, 2)
    y_batch = torch.tensor(y_np, dtype=torch.long, device=device)
    y_oh = F.one_hot(y_batch, num_classes=10).float()

    # Save old params
    old_params = {n: p.clone() for n, p in model.named_parameters()}

    model.train()
    output = model(x_batch, y_oh)

    optimizer.zero_grad()
    output.sum().backward()

    # Check targets (stored in .grad) before optimizer step
    if step == 0:
        print("\n=== Targets (grad) after backward, step 0 ===")
        for name, p in model.named_parameters():
            if p.grad is not None:
                pseudo_grad = p.data - p.grad  # this is what Muon will see
                print(f"  {name:40s}  param_norm={p.data.norm():.4f}  target_norm={p.grad.norm():.4f}  pseudo_grad_norm={pseudo_grad.norm():.4f}  pseudo_grad/param={pseudo_grad.norm()/max(p.data.norm(), 1e-8):.4f}")
            else:
                print(f"  {name:40s}  NO GRAD")

    optimizer.step()

    # Check param changes
    if step == 0:
        print("\n=== Param changes after step 0 ===")
        for name, p in model.named_parameters():
            delta = (p - old_params[name]).norm()
            print(f"  {name:40s}  delta_norm={delta:.6f}  param_norm={p.data.norm():.4f}")

    # Eval
    with torch.no_grad():
        logits = model.forward_eval(x_batch)
    ce_loss = float(F.cross_entropy(logits, y_batch))
    train_acc = float((logits.argmax(dim=-1) == y_batch).float().mean())
    print(f"Step {step:3d}  loss={ce_loss:.4f}  train_acc={train_acc:.4f}")
