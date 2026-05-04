"""Diagnostic 19: Combined best settings - long run."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'frameworks')))

import torch
import torch.nn.functional as F
import ptorch.optim_static as ptorch_optim_static
import ptorch.config as ptorch_config
from ptorch.core.overrides import apply_overrides

apply_overrides()
ptorch_config.use_projections = True
ptorch_config.config.update("frozen_a_weights", True)
ptorch_config.config.update("muon_weights", False)
ptorch_config.config.update("muon_activations_lr", 0.5)

from experiments.cnn_benchmarks.bench_ptorch_resnet import ResNet8_PTorch
from experiments.shared.data import InfiniteCifarDataModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

torch.manual_seed(42)
model = ResNet8_PTorch(classes=10, in_channels=3).to(device)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
ds = InfiniteCifarDataModule(batch_size=256, data_dir="./dataset", seed=42)
train_iter = ds.train_iterator()
val_loader = ds.val_dataloader()
test_loader = ds.test_dataloader()
optimizer = ptorch_optim_static.ProjectionSGD(model.parameters(), lr=0.03)

def eval_acc(loader):
    model.eval()
    accs = []
    with torch.no_grad():
        for xv, yv in loader:
            xv = torch.tensor(xv, dtype=torch.float32, device=device)
            if xv.shape[-1] in [1, 3]:
                xv = xv.permute(0, 3, 1, 2)
            yv = torch.tensor(yv, dtype=torch.long, device=device)
            logits = model.forward_eval(xv)
            accs.append((logits.argmax(dim=-1) == yv).float().mean())
    model.train()
    return float(torch.stack(accs).mean())

best_val = 0
best_step = 0
best_state = None

for step in range(10000):
    x_np, y_np = next(train_iter)
    x_batch = torch.tensor(x_np, dtype=torch.float32, device=device)
    if x_batch.shape[-1] in [1, 3]:
        x_batch = x_batch.permute(0, 3, 1, 2)
    y_batch = torch.tensor(y_np, dtype=torch.long, device=device)
    y_oh = F.one_hot(y_batch, num_classes=10).float()

    model.train()
    output = model(x_batch, y_oh)
    optimizer.zero_grad()
    output.sum().backward()
    optimizer.step()

    if step % 500 == 0 or step == 9999:
        with torch.no_grad():
            logits = model.forward_eval(x_batch)
        ce_loss = float(F.cross_entropy(logits, y_batch))
        train_acc = float((logits.argmax(dim=-1) == y_batch).float().mean())
        feat = model._forward_features(x_batch)
        feat_std = float(feat.detach().std())
        val = eval_acc(val_loader)
        if val > best_val:
            best_val = val
            best_step = step
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  step={step:5d}  loss={ce_loss:.4f}  acc={train_acc:.4f}  val={val:.4f}  best={best_val:.4f}@{best_step}  feat_std={feat_std:.4f}")

if best_state:
    model.load_state_dict(best_state)
test_acc = eval_acc(test_loader)
print(f"\nFinal: best_val={best_val:.4f}@step{best_step}  test_acc={test_acc:.4f}")
