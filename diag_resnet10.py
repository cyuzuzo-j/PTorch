"""Diagnostic 10: Test Conv2D without ConvPatchProjection averaging.
Monkey-patch Conv2D.forward to always use standard unfold (standard backward
through fold), while keeping projection-based matmul for weights.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'frameworks')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import ptorch.optim_static as ptorch_optim_static
import ptorch.nn.modules as pnn
import ptorch.config as ptorch_config
from ptorch.core.overrides import apply_overrides

apply_overrides()
ptorch_config.use_projections = True

from experiments.cnn_benchmarks.bench_ptorch_resnet import ResNet8_PTorch
from experiments.shared.data import InfiniteCifarDataModule

# Monkey-patch Conv2D.forward to skip ConvPatchProjection
_original_conv2d_forward = pnn.Conv2D.forward

def patched_conv2d_forward(self, input):
    padding = self._resolve_padding(input.shape[2], input.shape[3])
    # Always use standard unfold — no contractive averaging in backward
    patches = F.unfold(input, self.kernel_size, dilation=1, padding=padding, stride=self.stride)
    kH, kW = self.kernel_size
    sH, sW = self.stride
    if isinstance(padding, tuple):
        pad_h, pad_w = padding
    else:
        pad_h = pad_w = padding
    H_out = (input.shape[2] + 2 * pad_h - kH) // sH + 1
    W_out = (input.shape[3] + 2 * pad_w - kW) // sW + 1
    patches = patches.view(input.shape[0], -1, H_out, W_out).permute(0, 2, 3, 1)
    # Matmul through Linear — still uses projections
    out = self.linear(patches)
    return out.permute(0, 3, 1, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

configs = [
    ("With ConvPatchProjection (default)", False),
    ("Without ConvPatchProjection (standard unfold)", True),
]

for name, use_patch in configs:
    if use_patch:
        pnn.Conv2D.forward = patched_conv2d_forward
    else:
        pnn.Conv2D.forward = _original_conv2d_forward

    print(f"\n=== {name} ===")
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

pnn.Conv2D.forward = _original_conv2d_forward  # restore
