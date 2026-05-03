"""Diagnostic 8: test muon_activations_scale and muon_activations off."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'frameworks')))

import torch
import torch.nn.functional as F
import ptorch.optim_static as ptorch_optim_static
import ptorch.config as ptorch_config
from ptorch.core.overrides import apply_overrides

apply_overrides()

from experiments.cnn_benchmarks.bench_ptorch_resnet import ResNet8_PTorch
from experiments.shared.data import InfiniteCifarDataModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

configs = [
    ("muon_act=T, scale=F (default)", True, False),
    ("muon_act=T, scale=T", True, True),
    ("muon_act=F", False, False),
]

for name, muon_act, muon_scale in configs:
    print(f"\n=== {name} ===")
    ptorch_config.use_projections = True
    ptorch_config.muon_activations = muon_act
    ptorch_config.muon_activations_scale = muon_scale

    torch.manual_seed(42)
    model = ResNet8_PTorch(classes=10, in_channels=3).to(device)
    ds = InfiniteCifarDataModule(batch_size=256, data_dir="./dataset", seed=42)
    train_iter = ds.train_iterator()
    val_loader = ds.val_dataloader()
    optimizer = ptorch_optim_static.AlternatingProjections(model.parameters(), lr=1.0)

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

        if step % 100 == 0 or step == 499:
            with torch.no_grad():
                logits = model.forward_eval(x_batch)
            ce_loss = float(F.cross_entropy(logits, y_batch))
            train_acc = float((logits.argmax(dim=-1) == y_batch).float().mean())
            feat = model._forward_features(x_batch)
            feat_std = float(feat.detach().std())
            val = eval_acc()
            print(f"  step={step:4d}  loss={ce_loss:.4f}  acc={train_acc:.4f}  val={val:.4f}  feat_std={feat_std:.4f}")

ptorch_config.muon_activations = True
ptorch_config.muon_activations_scale = False
