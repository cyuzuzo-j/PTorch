"""Diagnostic 23: Untried hyperparameters with frozen_a_weights=True.

Tests things never explored in diag_resnet1-22:
  C0: Baseline (reproduce ~50%) — 10K steps, no aug, SGD lr=0.03
  C1: Longer training — 30K steps
  C2: Augmentation — flip + translate(4)
  C3: Longer + Aug — 30K steps + augmentation
  C4: Cosine LR — cosine annealing 0.03→0.001 over 30K + aug
  C5: CE lambda=5 — stronger cross-entropy target push, 30K + aug
  C6: More Newton iters — num_iters=30 for Conv2D, 30K + aug
"""
import sys, os, math
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

from experiments.shared.data import InfiniteCifarDataModule
from experiments.shared.data import InfiniteCifarLoader, FiniteCifarLoader

apply_overrides()
ptorch_config.use_projections = True
ptorch_config.config.update("frozen_a_weights", True)
ptorch_config.config.update("muon_activations_lr", 0.5)
ptorch_config.config.update("muon_activations_norm_preserve", False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

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

# ── Model ────────────────────────────────────────────────────────────────────

class BasicBlock(tnn.Module):
    def __init__(self, in_channels, out_channels, stride=1, alpha=1.0, g=1.0, num_iters=15):
        super().__init__()
        self.conv1 = pnn.Conv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, alpha=alpha, g=g, num_iters=num_iters)
        self.relu1 = pnn.LeakyReLU(0.1)
        self.conv2 = pnn.Conv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1, alpha=alpha, g=g, num_iters=num_iters)
        self.relu2 = pnn.LeakyReLU(0.1)
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = pnn.Conv2D(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, alpha=alpha, g=g, num_iters=num_iters)
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


class CrossEntropyLambda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, num_steps, lmbda):
        ctx.save_for_backward(logits, labels)
        ctx.num_steps = num_steps
        ctx.lmbda = lmbda
        return logits

    @staticmethod
    def backward(ctx, z_target):
        logits, labels = ctx.saved_tensors
        x = logits
        for _ in range(ctx.num_steps):
            x = x + ctx.lmbda * (labels - F.softmax(x, dim=-1))
        return x, labels, None, None


class ResNet8(tnn.Module):
    def __init__(self, classes=10, in_channels=3, alpha=1.0, g=1.0, num_iters=15, ce_lambda=1.0):
        super().__init__()
        self.block1 = BasicBlock(in_channels, 128, stride=2, alpha=alpha, g=g, num_iters=num_iters)
        self.block2 = BasicBlock(128, 512, stride=2, alpha=alpha, g=g, num_iters=num_iters)
        self.block3 = BasicBlock(512, 1024, stride=2, alpha=alpha, g=g, num_iters=num_iters)
        self.head = pnn.Linear(1024, classes, norm="inf")
        self.ce_lambda = ce_lambda

    def _forward_features(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = ProjectedGlobalAvgPool.apply(x)
        return x

    def forward(self, x, y_oh):
        x = self._forward_features(x)
        logits = self.head(x)
        return CrossEntropyLambda.apply(logits, y_oh, 5, self.ce_lambda)

    def forward_eval(self, x):
        x = self._forward_features(x)
        return self.head(x)


# ── Data ─────────────────────────────────────────────────────────────────────

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

def train_config(name, model, ds, opt_lr, max_steps, lr_schedule=None):
    print(f"\n{'='*60}\n  {name}\n{'='*60}", flush=True)

    model = model.to(device)
    train_iter = ds.train_iterator()
    val_loader = ds.val_dataloader()
    optimizer = ptorch_optim_static.ProjectionSGD(model.parameters(), lr=opt_lr)

    if lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=0.001)
    else:
        scheduler = None

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
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if step % 500 == 0 or step == max_steps - 1:
            with torch.no_grad():
                logits = model.forward_eval(x_batch)
            ce_loss = float(F.cross_entropy(logits, y_batch))
            train_acc = float((logits.argmax(dim=-1) == y_batch).float().mean())
            feat = model._forward_features(x_batch)
            feat_std = float(feat.detach().std())

            val = eval_acc()
            best_val = max(best_val, val)
            cur_lr = scheduler.get_last_lr()[0] if scheduler else opt_lr

            elapsed = time.time() - t0
            print(f"  step={step:5d}  loss={ce_loss:.4f}  acc={train_acc:.4f}  "
                  f"val={val:.4f}  best={best_val:.4f}  feat_std={feat_std:.4f}  "
                  f"lr={cur_lr:.5f}  time={elapsed:.0f}s", flush=True)

    return best_val


# ── Configs ──────────────────────────────────────────────────────────────────

configs = [
    # (name, max_steps, aug, opt_lr, lr_schedule, ce_lambda, num_iters)
    ("C0: baseline 10K",         10000, False, 0.03, None,     1.0, 15),
    ("C1: 30K steps",            30000, False, 0.03, None,     1.0, 15),
    ("C2: augmentation",         10000, True,  0.03, None,     1.0, 15),
    ("C3: 30K + aug",            30000, True,  0.03, None,     1.0, 15),
    ("C4: cosine LR + 30K + aug",30000, True,  0.03, "cosine", 1.0, 15),
    ("C5: CE lambda=5 + 30K+aug",30000, True,  0.03, "cosine", 5.0, 15),
    ("C6: newton=30 + 30K + aug", 30000, True,  0.03, "cosine", 1.0, 30),
]

results = {}
for (name, max_steps, aug, opt_lr, lr_schedule, ce_lambda, num_iters) in configs:
    torch.manual_seed(42)
    model = ResNet8(classes=10, in_channels=3, num_iters=num_iters, ce_lambda=ce_lambda)

    if aug:
        ds = AugmentedCifarDataModule(batch_size=256, data_dir="./dataset", seed=42)
    else:
        ds = InfiniteCifarDataModule(batch_size=256, data_dir="./dataset", seed=42)

    try:
        best = train_config(name, model, ds, opt_lr, max_steps, lr_schedule)
        results[name] = best
        print(f"  >>> RESULT: {name} = {best:.4f}", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        results[name] = -1.0
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\n{'='*60}")
print("  SUMMARY")
print(f"{'='*60}")
for name, acc in results.items():
    print(f"  {name:40s}  val={acc:.4f}")
