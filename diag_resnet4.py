"""Diagnostic 4: longer training run with AlternatingProjections."""
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

from experiments.cnn_benchmarks.bench_ptorch_resnet import ResNet8_PTorch
from experiments.shared.data import InfiniteCifarDataModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for opt_name, opt_kwargs in [
    ("AlternatingProjections", {"lr": 1.0}),
    ("ProjectionSGD", {"lr": 1.0, "momentum": 0.9}),
]:
    print(f"\n=== {opt_name} {opt_kwargs} ===")
    torch.manual_seed(42)
    model = ResNet8_PTorch(classes=10, in_channels=3).to(device)
    ds = InfiniteCifarDataModule(batch_size=128, data_dir="./dataset", seed=42)
    train_iter = ds.train_iterator()

    opt_cls = getattr(ptorch_optim_static, opt_name)
    optimizer = opt_cls(model.parameters(), **opt_kwargs)

    for step in range(500):
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

        if step % 50 == 0 or step == 499:
            with torch.no_grad():
                logits = model.forward_eval(x_batch)
            ce_loss = float(F.cross_entropy(logits, y_batch))
            train_acc = float((logits.argmax(dim=-1) == y_batch).float().mean())
            feat = model._forward_features(x_batch)
            feat_std = float(feat.detach().std())
            print(f"  step={step:4d}  loss={ce_loss:.4f}  acc={train_acc:.4f}  feat_std={feat_std:.4f}")
