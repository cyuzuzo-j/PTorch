"""Diagnostic 24: Optimizer & regularization (5K steps, frozen_a=True).

  C0: Baseline SGD lr=0.03
  C1: SGD lr=0.03 momentum=0.9
  C2: SGD lr=0.1
  C3: SGD lr=0.3
  C4: SGD lr=0.03 weight_decay=1e-4
  C5: SGD lr=0.03 weight_decay=1e-3
  C6: SGD lr=0.1 momentum=0.9 weight_decay=1e-4
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'frameworks')))

import torch, gc, time
import torch.nn as tnn
import torch.nn.functional as F
import ptorch.optim_static as ptorch_optim_static
import ptorch.nn.modules as pnn
import ptorch.config as ptorch_config
from ptorch.core.overrides import apply_overrides
from ptorch.core.ops import process_activation_target
from experiments.shared.data import InfiniteCifarDataModule

apply_overrides()
ptorch_config.use_projections = True
ptorch_config.config.update("frozen_a_weights", True)
ptorch_config.config.update("muon_activations_lr", 0.5)
ptorch_config.config.update("muon_activations_norm_preserve", False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}", flush=True)

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
        return (process_activation_target(skip, z_target - out),
                process_activation_target(out, z_target - skip))

class BasicBlock(tnn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = pnn.Conv2D(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.relu1 = pnn.LeakyReLU(0.1)
        self.conv2 = pnn.Conv2D(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.relu2 = pnn.LeakyReLU(0.1)
        self.shortcut = pnn.Conv2D(in_ch, out_ch, kernel_size=1, stride=stride, padding=0) if stride != 1 or in_ch != out_ch else None
        with torch.no_grad():
            self.conv2.linear.weight.mul_(0.0)

    def forward(self, x):
        conv_in = x.detach().requires_grad_(True)
        out = self.relu1(self.conv1(conv_in))
        out = self.conv2(out)
        skip = self.shortcut(x) if self.shortcut is not None else x
        return self.relu2(ResidualAdd.apply(skip, out))

class ResNet8(tnn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = BasicBlock(3, 128, stride=2)
        self.block2 = BasicBlock(128, 512, stride=2)
        self.block3 = BasicBlock(512, 1024, stride=2)
        self.head = pnn.Linear(1024, 10, norm="inf")
        self.loss = pnn.CrossEntropy()

    def _features(self, x):
        return ProjectedGlobalAvgPool.apply(self.block3(self.block2(self.block1(x))))

    def forward(self, x, y_oh):
        return self.loss(self.head(self._features(x)), y_oh)

    def forward_eval(self, x):
        return self.head(self._features(x))

def run(name, opt_kwargs, max_steps=5000):
    print(f"\n{'='*60}\n  {name}\n{'='*60}", flush=True)
    torch.manual_seed(42)
    model = ResNet8().to(device)
    ds = InfiniteCifarDataModule(batch_size=256, data_dir="./dataset", seed=42)
    train_iter = ds.train_iterator()
    val_loader = ds.val_dataloader()
    optimizer = ptorch_optim_static.ProjectionSGD(model.parameters(), **opt_kwargs)

    def eval_acc():
        model.eval()
        accs = []
        with torch.no_grad():
            for xv, yv in val_loader:
                xv = torch.tensor(xv, dtype=torch.float32, device=device)
                if xv.shape[-1] in [1, 3]: xv = xv.permute(0, 3, 1, 2)
                yv = torch.tensor(yv, dtype=torch.long, device=device)
                accs.append((model.forward_eval(xv).argmax(-1) == yv).float().mean())
        model.train()
        return float(torch.stack(accs).mean())

    best_val = 0; t0 = time.time()
    for step in range(max_steps):
        x_np, y_np = next(train_iter)
        x = torch.tensor(x_np, dtype=torch.float32, device=device)
        if x.shape[-1] in [1, 3]: x = x.permute(0, 3, 1, 2)
        y = torch.tensor(y_np, dtype=torch.long, device=device)
        y_oh = F.one_hot(y, 10).float()
        model.train()
        model(x, y_oh).sum().backward()
        optimizer.step(); optimizer.zero_grad()

        if step % 1000 == 0 or step == max_steps - 1:
            with torch.no_grad(): logits = model.forward_eval(x)
            val = eval_acc(); best_val = max(best_val, val)
            print(f"  step={step:5d}  loss={float(F.cross_entropy(logits, y)):.4f}  "
                  f"val={val:.4f}  best={best_val:.4f}  time={time.time()-t0:.0f}s", flush=True)
    return best_val

configs = [
    ("C0: baseline lr=0.03",             dict(lr=0.03)),
    ("C1: lr=0.03 mom=0.9",              dict(lr=0.03, momentum=0.9)),
    ("C2: lr=0.1",                        dict(lr=0.1)),
    ("C3: lr=0.3",                        dict(lr=0.3)),
    ("C4: lr=0.03 wd=1e-4",              dict(lr=0.03, weight_decay=1e-4)),
    ("C5: lr=0.03 wd=1e-3",              dict(lr=0.03, weight_decay=1e-3)),
    ("C6: lr=0.1 mom=0.9 wd=1e-4",       dict(lr=0.1, momentum=0.9, weight_decay=1e-4)),
]

results = {}
for name, kw in configs:
    try:
        results[name] = run(name, kw)
        print(f"  >>> RESULT: {name} = {results[name]:.4f}", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        results[name] = -1.0
    gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

print(f"\n{'='*60}\n  SUMMARY (diag_resnet24)\n{'='*60}")
for n, a in results.items():
    print(f"  {n:40s}  val={a:.4f}")
