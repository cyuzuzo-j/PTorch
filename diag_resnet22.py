"""Diagnostic 22: Push accuracy without FrozenA.

Tests structural fixes:
1. Higher g for stronger constraint satisfaction → meaningful weight signal
2. Muon weights with scale for adaptive weight step sizing
3. Muon activations with scale for adaptive activation steps (fix shrinkage)
4. Data augmentation (flip + translate)
5. Remove detach + AverageGradient for better target propagation between blocks
"""
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
from ptorch.core.ops import process_activation_target, AverageGradient

from experiments.shared.data import InfiniteCifarDataModule
from experiments.shared.data import InfiniteCifarLoader, FiniteCifarLoader

apply_overrides()
ptorch_config.use_projections = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Projection-aware global average pooling ──────────────────────────────────

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

# ── Residual addition ────────────────────────────────────────────────────────

class ResidualAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, skip, out):
        ctx.save_for_backward(skip, out)
        return skip + out

    @staticmethod
    def backward(ctx, z_target):
        skip, out = ctx.saved_tensors
        return (process_activation_target(skip, z_target - out),
                process_activation_target(out, z_target - skip))

# ── BasicBlock variants ──────────────────────────────────────────────────────

class BasicBlock(tnn.Module):
    """Standard block with detach (skip-only target propagation)."""
    def __init__(self, in_channels, out_channels, stride=1, alpha=1.0, g=1.0):
        super().__init__()
        self.conv1 = pnn.Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, alpha=alpha, g=g)
        self.relu1 = pnn.LeakyReLU(0.1)
        self.conv2 = pnn.Conv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1, alpha=alpha, g=g)
        self.relu2 = pnn.LeakyReLU(0.1)
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = pnn.Conv2D(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, alpha=alpha, g=g)
        with torch.no_grad():
            self.conv2.linear.weight.mul_(0.0)

    def forward(self, x):
        conv_in = x.detach().requires_grad_(True)
        out = self.relu1(self.conv1(conv_in))
        out = self.conv2(out)
        skip = self.shortcut(x) if self.shortcut is not None else x
        x = ResidualAdd.apply(skip, out)
        x = self.relu2(x)
        return x


class BasicBlockAvgGrad(tnn.Module):
    """Block WITHOUT detach — both branches propagate targets via AverageGradient."""
    def __init__(self, in_channels, out_channels, stride=1, alpha=1.0, g=1.0):
        super().__init__()
        self.conv1 = pnn.Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, alpha=alpha, g=g)
        self.relu1 = pnn.LeakyReLU(0.1)
        self.conv2 = pnn.Conv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1, alpha=alpha, g=g)
        self.relu2 = pnn.LeakyReLU(0.1)
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = pnn.Conv2D(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, alpha=alpha, g=g)
        with torch.no_grad():
            self.conv2.linear.weight.mul_(0.0)

    def forward(self, x):
        x_split = AverageGradient.apply(x, 2)
        out = self.relu1(self.conv1(x_split))
        out = self.conv2(out)
        skip = self.shortcut(x_split) if self.shortcut is not None else x_split
        x = ResidualAdd.apply(skip, out)
        x = self.relu2(x)
        return x


# ── Model ────────────────────────────────────────────────────────────────────

class ResNet8(tnn.Module):
    def __init__(self, classes=10, in_channels=3, alpha=1.0, g=1.0, block_cls=BasicBlock):
        super().__init__()
        self.block1 = block_cls(in_channels, 128, stride=2, alpha=alpha, g=g)
        self.block2 = block_cls(128, 512, stride=2, alpha=alpha, g=g)
        self.block3 = block_cls(512, 1024, stride=2, alpha=alpha, g=g)
        self.head = pnn.Linear(1024, classes, norm="inf")
        self.loss = pnn.CrossEntropy()

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
        return self.head(x)


# ── Data with augmentation ───────────────────────────────────────────────────

class AugmentedCifarDataModule:
    def __init__(self, batch_size=256, seed=42, data_dir="./dataset"):
        self.batch_size = batch_size
        self.seed = seed
        self.data_dir = data_dir

    def train_iterator(self):
        loader = InfiniteCifarLoader(
            self.data_dir, train=True, batch_size=self.batch_size,
            aug={'flip': True, 'translate': 4},
            aug_seed=self.seed, order_seed=self.seed)
        def _iter():
            for _, x, y in loader:
                yield x.float().permute(0, 2, 3, 1).cpu().numpy(), y.cpu().numpy()
        return _iter()

    def val_dataloader(self):
        loader = InfiniteCifarLoader(self.data_dir, train=False, batch_size=self.batch_size)
        return FiniteCifarLoader(loader, 10000)


# ── Training loop ────────────────────────────────────────────────────────────

def train_config(name, model, ds, opt_lr, max_steps=10000):
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    snap = ptorch_config.config.snapshot()
    print(f"  Config: frozen_a={snap['frozen_a_weights']}, "
          f"muon_w={snap['muon_weights']}, "
          f"muon_w_scale={snap.get('muon_weights_scale', False)}, "
          f"muon_a_scale={snap.get('muon_activations_scale', False)}, "
          f"muon_a_lr={snap['muon_activations_lr']}, "
          f"proj_g={snap['projection_g']}")

    model = model.to(device)
    train_iter = ds.train_iterator()
    val_loader = ds.val_dataloader()
    optimizer = ptorch_optim_static.ProjectionSGD(model.parameters(), lr=opt_lr)

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
    import time
    t0 = time.time()

    for step in range(max_steps):
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

        is_eval_step = (step % 500 == 0 or step == max_steps - 1)
        if is_eval_step:
            # Measure pseudo-gradient norms BEFORE optimizer step
            # p.grad = p_proj (projection target), so p - p_proj = pseudo-gradient
            w_norms = []
            for n, p in model.named_parameters():
                if 'weight' in n and p.grad is not None:
                    pg_norm = float((p.data - p.grad).norm())
                    p_norm = float(p.data.norm())
                    w_norms.append(pg_norm / (p_norm + 1e-8))
            w_change = sum(w_norms) / max(len(w_norms), 1)

        optimizer.step()

        if is_eval_step:
            with torch.no_grad():
                logits = model.forward_eval(x_batch)
            ce_loss = float(F.cross_entropy(logits, y_batch))
            train_acc = float((logits.argmax(dim=-1) == y_batch).float().mean())
            feat = model._forward_features(x_batch)
            feat_std = float(feat.detach().std())

            val = eval_acc()
            best_val = max(best_val, val)

            elapsed = time.time() - t0
            print(f"  step={step:5d}  loss={ce_loss:.4f}  acc={train_acc:.4f}  "
                  f"val={val:.4f}  best={best_val:.4f}  feat_std={feat_std:.4f}  "
                  f"w_change={w_change:.6f}  time={elapsed:.0f}s")

    return best_val


# ── Configs ──────────────────────────────────────────────────────────────────

configs = [
    # (name, frozen_a, muon_w, muon_w_scale, muon_w_lr, muon_a_scale, muon_a_lr, proj_g, opt_lr, block_cls, aug)
    # C5: C4 + remove detach (AverageGradient block)
    ("C1.5: C1 + avg_grad_block", True, False, False, 0.02, False, 0.5, 1.0, 0.03, BasicBlockAvgGrad, False),

    # C1: No FrozenA baseline (joint projection, default g=1)
    ("C1: no_frozen, g=1 baseline", True, False, False, 0.02, False, 0.5, 1.0, 0.03, BasicBlock, False),

    # C2: No FrozenA, higher g for stronger constraint signal
    ("C2: no_frozen, g=5", True, False, False, 0.02, False, 0.5, 5.0, 0.03, BasicBlock, False),

    # C3: No FrozenA, higher g + Muon weights w/ scale
    ("C3: no_frozen, g=5, muon_w+scale", True, True, True, 0.1, False, 0.5, 5.0, 0.1, BasicBlock, False),

    # C4: No FrozenA, g=5, muon_w+scale, muon_a+scale (fix shrinkage)
    ("C4: C3 + muon_a_scale", True, True, True, 0.1, True, 0.5, 5.0, 0.1, BasicBlock, False),
    
    # C5: C4 + remove detach (AverageGradient block)
    ("C5: C4 + avg_grad_block", True, True, True, 0.1, True, 0.5, 5.0, 0.1, BasicBlockAvgGrad, False),

    # C6: C5 + data augmentation
    ("C6: C5 + augmentation", True, True, True, 0.1, True, 0.5, 5.0, 0.1, BasicBlockAvgGrad, True),
]

results = {}
for (name, frozen_a, muon_w, muon_w_scale, muon_w_lr,
     muon_a_scale, muon_a_lr, proj_g, opt_lr, block_cls, aug) in configs:

    ptorch_config.config.update("frozen_a_weights", frozen_a)
    ptorch_config.config.update("muon_weights", muon_w)
    ptorch_config.config.update("muon_weights_scale", muon_w_scale)
    ptorch_config.config.update("muon_weights_lr", muon_w_lr)
    ptorch_config.config.update("muon_activations", True)
    ptorch_config.config.update("muon_activations_scale", muon_a_scale)
    ptorch_config.config.update("muon_activations_lr", muon_a_lr)
    ptorch_config.config.update("muon_activations_norm_preserve", False)
    ptorch_config.config.update("projection_g", proj_g)

    torch.manual_seed(42)
    model = ResNet8(classes=10, in_channels=3, block_cls=block_cls)

    if aug:
        ds = AugmentedCifarDataModule(batch_size=256, data_dir="./dataset", seed=42)
    else:
        ds = InfiniteCifarDataModule(batch_size=256, data_dir="./dataset", seed=42)

    try:
        best = train_config(name, model, ds, opt_lr, max_steps=7500)
        results[name] = best
    except Exception as e:
        print(f"  FAILED: {e}")
        results[name] = -1.0
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\n\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for name, val in results.items():
    print(f"  {val:.4f}  {name}")
