"""Diagnostic 14: FrozenA weight targets + ProjectionSGD to control step size."""
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

configs = [
    ("Baseline AP", False, "AlternatingProjections", {"lr": 1.0}),
    ("FrozenA + AP", True, "AlternatingProjections", {"lr": 1.0}),
    ("FrozenA + SGD lr=0.3", True, "ProjectionSGD", {"lr": 0.3}),
    ("FrozenA + SGD lr=0.1", True, "ProjectionSGD", {"lr": 0.1}),
    ("FrozenA + SGD lr=0.05", True, "ProjectionSGD", {"lr": 0.05}),
    ("FrozenA + SGD lr=0.01", True, "ProjectionSGD", {"lr": 0.01}),
]

for name, frozen_a, opt_name, opt_kwargs in configs:
    print(f"\n=== {name} ===")
    ptorch_config.config.update("frozen_a_weights", frozen_a)
    ptorch_config.config.update("muon_weights", False)

    torch.manual_seed(42)
    model = ResNet8_PTorch(classes=10, in_channels=3).to(device)
    ds = InfiniteCifarDataModule(batch_size=256, data_dir="./dataset", seed=42)
    train_iter = ds.train_iterator()
    val_loader = ds.val_dataloader()
    opt_cls = getattr(ptorch_optim_static, opt_name)
    optimizer = opt_cls(model.parameters(), **opt_kwargs)

    init_norms = {n: p.detach().norm().item() for n, p in model.named_parameters() if 'weight' in n}

    def eval_acc():
        model.eval()
        accs = []
        with torch.no_grad():
            for xv, yv in val_loader:
                xv = torch.tensor(xv, dtype=torch.float32, device=device)
                if xv.shape[-1] in [1, 3]:
                    xv = xv.permute(0, 3, 1, 2)
                yv = torch.tensor(yv, dtype=torch.long, device=device)
                logits = model.forward_eval(xv)
                accs.append((logits.argmax(dim=-1) == yv).float().mean())
        model.train()
        return float(torch.stack(accs).mean())

    for step in range(1000):
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

        if step % 200 == 0 or step == 999:
            with torch.no_grad():
                logits = model.forward_eval(x_batch)
            ce_loss = float(F.cross_entropy(logits, y_batch))
            train_acc = float((logits.argmax(dim=-1) == y_batch).float().mean())
            feat = model._forward_features(x_batch)
            feat_std = float(feat.detach().std())
            val = eval_acc()
            print(f"  step={step:4d}  loss={ce_loss:.4f}  acc={train_acc:.4f}  val={val:.4f}  feat_std={feat_std:.4f}")

ptorch_config.config.update("frozen_a_weights", True)
