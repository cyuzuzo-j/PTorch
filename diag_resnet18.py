"""Diagnostic 18: Test conv2 init strategy + remove detach."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'frameworks')))

import torch
import torch.nn as tnn
import torch.nn.functional as F
import ptorch.optim_static as ptorch_optim_static
import ptorch.nn.modules as pnn
import ptorch.config as ptorch_config
from ptorch.core.overrides import apply_overrides
from ptorch.core.ops import process_activation_target
from ptorch.nn.modules import HardMarginLoss

apply_overrides()
ptorch_config.use_projections = True
ptorch_config.config.update("frozen_a_weights", True)
ptorch_config.config.update("muon_weights", False)


class ProjectedGlobalAvgPool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.mean(dim=[2, 3])

    @staticmethod
    def backward(ctx, z_target):
        x, = ctx.saved_tensors
        N = x.shape[2] * x.shape[3]
        mean_x = x.mean(dim=[2, 3], keepdim=True)
        z_exp = z_target.unsqueeze(-1).unsqueeze(-1)
        correction = (z_exp - mean_x) / (N + 1)
        return process_activation_target(x, x + correction)


class ResidualAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, skip, out):
        ctx.save_for_backward(skip, out)
        return skip + out

    @staticmethod
    def backward(ctx, z_target):
        skip, out = ctx.saved_tensors
        return process_activation_target(skip, z_target - out), process_activation_target(out, z_target - skip)


class BasicBlock(tnn.Module):
    def __init__(self, in_channels, out_channels, stride=1, alpha=1.0, g=1.0, zero_init=True, use_detach=True):
        super().__init__()
        self.conv1 = pnn.Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, alpha=alpha, g=g)
        self.relu1 = pnn.LeakyReLU(0.1)
        self.conv2 = pnn.Conv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1, alpha=alpha, g=g)
        self.relu2 = pnn.LeakyReLU(0.1)
        self.use_detach = use_detach

        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = pnn.Conv2D(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, alpha=alpha, g=g)

        if zero_init:
            with torch.no_grad():
                self.conv2.linear.weight.mul_(0.0)

    def forward(self, x):
        if self.use_detach:
            conv_in = x.detach().requires_grad_(True)
        else:
            conv_in = x
        out = self.relu1(self.conv1(conv_in))
        out = self.conv2(out)

        if self.shortcut is not None:
            skip = self.shortcut(x)
        else:
            skip = x

        x = ResidualAdd.apply(skip, out)
        x = self.relu2(x)
        return x


class ResNet8(tnn.Module):
    def __init__(self, classes=10, in_channels=3, alpha=1.0, g=1.0, zero_init=True, use_detach=True):
        super().__init__()
        self.block1 = BasicBlock(in_channels, 16, stride=2, alpha=alpha, g=g, zero_init=zero_init, use_detach=use_detach)
        self.block2 = BasicBlock(16, 32, stride=2, alpha=alpha, g=g, zero_init=zero_init, use_detach=use_detach)
        self.block3 = BasicBlock(32, 64, stride=2, alpha=alpha, g=g, zero_init=zero_init, use_detach=use_detach)

        self.head = pnn.Linear(64, classes, norm="inf")
        self.loss = HardMarginLoss()

    def _forward_features(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = ProjectedGlobalAvgPool.apply(x)
        return x

    def forward(self, x, y_oh):
        x = self._forward_features(x)
        logits = self.head(x)
        return self.loss(logits, y_oh)

    def forward_eval(self, x):
        x = self._forward_features(x)
        logits = self.head(x)
        return logits


from experiments.shared.data import InfiniteCifarDataModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

configs = [
    ("zero_init=True detach=True (current)", True, True),
    ("zero_init=False detach=True", False, True),
    ("zero_init=True detach=False", True, False),
    ("zero_init=False detach=False", False, False),
]

for name, zero_init, use_detach in configs:
    print(f"\n=== {name} ===")
    torch.manual_seed(42)
    model = ResNet8(classes=10, in_channels=3, zero_init=zero_init, use_detach=use_detach).to(device)
    ds = InfiniteCifarDataModule(batch_size=256, data_dir="./dataset", seed=42)
    train_iter = ds.train_iterator()
    val_loader = ds.val_dataloader()
    optimizer = ptorch_optim_static.ProjectionSGD(model.parameters(), lr=0.03)

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

    best_val = 0
    for step in range(3000):
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

        if step % 500 == 0 or step == 2999:
            with torch.no_grad():
                logits = model.forward_eval(x_batch)
            ce_loss = float(F.cross_entropy(logits, y_batch))
            train_acc = float((logits.argmax(dim=-1) == y_batch).float().mean())
            feat = model._forward_features(x_batch)
            feat_std = float(feat.detach().std())
            val = eval_acc()
            best_val = max(best_val, val)
            print(f"  step={step:4d}  loss={ce_loss:.4f}  acc={train_acc:.4f}  val={val:.4f}  best={best_val:.4f}  feat_std={feat_std:.4f}")
