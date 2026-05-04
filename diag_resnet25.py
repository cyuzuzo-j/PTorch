"""Diagnostic 25: Architecture & training variations (5K steps, frozen_a=True).

  C0: Baseline 128/512/1024
  C1: Narrow 64/128/256
  C2: Uniform 256/256/256
  C3: 4-block 64/128/256/512
  C4: No zero-init on conv2
  C5: Augmentation (flip + translate)
  C6: CE lambda=5 (stronger target push)
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
from experiments.shared.data import InfiniteCifarDataModule, InfiniteCifarLoader, FiniteCifarLoader

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
    def __init__(self, in_ch, out_ch, stride=1, zero_init=True):
        super().__init__()
        self.conv1 = pnn.Conv2D(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.relu1 = pnn.LeakyReLU(0.1)
        self.conv2 = pnn.Conv2D(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.relu2 = pnn.LeakyReLU(0.1)
        self.shortcut = pnn.Conv2D(in_ch, out_ch, kernel_size=1, stride=stride, padding=0) if stride != 1 or in_ch != out_ch else None
        if zero_init:
            with torch.no_grad():
                self.conv2.linear.weight.mul_(0.0)

    def forward(self, x):
        conv_in = x.detach().requires_grad_(True)
        out = self.relu1(self.conv1(conv_in))
        out = self.conv2(out)
        skip = self.shortcut(x) if self.shortcut is not None else x
        return self.relu2(ResidualAdd.apply(skip, out))

class CrossEntropyLambda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, num_steps, lmbda):
        ctx.save_for_backward(logits, labels)
        ctx.num_steps = num_steps; ctx.lmbda = lmbda
        return logits
    @staticmethod
    def backward(ctx, z_target):
        logits, labels = ctx.saved_tensors
        x = logits
        for _ in range(ctx.num_steps):
            x = x + ctx.lmbda * (labels - F.softmax(x, dim=-1))
        return x, labels, None, None

class ResNetFlex(tnn.Module):
    def __init__(self, widths=(128, 512, 1024), zero_init=True, ce_lambda=1.0):
        super().__init__()
        self.blocks = tnn.ModuleList()
        ch = 3
        for w in widths:
            self.blocks.append(BasicBlock(ch, w, stride=2, zero_init=zero_init))
            ch = w
        self.head = pnn.Linear(ch, 10, norm="inf")
        self.ce_lambda = ce_lambda

    def _features(self, x):
        for b in self.blocks:
            x = b(x)
        return ProjectedGlobalAvgPool.apply(x)

    def forward(self, x, y_oh):
        logits = self.head(self._features(x))
        if self.ce_lambda == 1.0:
            from ptorch.core.ops import CrossEntropyProjection
            return CrossEntropyProjection.apply(logits, y_oh)
        return CrossEntropyLambda.apply(logits, y_oh, 5, self.ce_lambda)

    def forward_eval(self, x):
        return self.head(self._features(x))

class AugCifar:
    def __init__(self, batch_size=256, seed=42, data_dir="./dataset"):
        self.bs = batch_size; self.seed = seed; self.dir = data_dir
    def train_iterator(self):
        loader = InfiniteCifarLoader(self.dir, train=True, batch_size=self.bs,
            aug={'flip': True, 'translate': 4}, aug_seed=self.seed, order_seed=self.seed)
        def _it():
            for _, x, y in loader:
                yield x.float().permute(0, 2, 3, 1).cpu().numpy(), y.cpu().numpy()
        return _it()
    def val_dataloader(self):
        return FiniteCifarLoader(InfiniteCifarLoader(self.dir, train=False, batch_size=self.bs), 10000)

def run(name, model, ds, max_steps=5000):
    print(f"\n{'='*60}\n  {name}  params={sum(p.numel() for p in model.parameters()):,}\n{'='*60}", flush=True)
    model = model.to(device)
    train_iter = ds.train_iterator()
    val_loader = ds.val_dataloader()
    optimizer = ptorch_optim_static.ProjectionSGD(model.parameters(), lr=0.03)

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
    ("C1: narrow 64/128/256",      dict(widths=(64, 128, 256)), False),
    ("C2: uniform 256/256/256",    dict(widths=(256, 256, 256)), False),
    ("C3: 4-block 64/128/256/512", dict(widths=(64, 128, 256, 512)), False),
    ("C4: no zero-init",           dict(widths=(128, 512, 1024), zero_init=False), False),
    ("C5: augmentation",           dict(widths=(128, 512, 1024)), True),
    ("C6: CE lambda=5",            dict(widths=(128, 512, 1024), ce_lambda=5.0), False),
    ("C0: baseline 128/512/1024",  dict(widths=(128, 512, 1024)), False),

]

results = {}
for name, kw, aug in configs:
    torch.manual_seed(42)
    model = ResNetFlex(**kw)
    ds = AugCifar(batch_size=256, data_dir="./dataset", seed=42) if aug else InfiniteCifarDataModule(batch_size=256, data_dir="./dataset", seed=42)
    try:
        results[name] = run(name, model, ds)
        print(f"  >>> RESULT: {name} = {results[name]:.4f}", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        results[name] = -1.0
    gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

print(f"\n{'='*60}\n  SUMMARY (diag_resnet25)\n{'='*60}")
for n, a in results.items():
    print(f"  {n:40s}  val={a:.4f}")
