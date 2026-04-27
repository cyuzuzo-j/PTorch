"""
Debug script: inspect FX node names, test PropagateCache without wandb.
Run from repo root: python3 experiments/boolean/debug_cache.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
import torch.nn as tnn
import torch.fx
from ptorch.nn.modules import Linear, LinearFrozen, LeakyReLU, ProjectionModule, HardMarginLoss
import ptorch.optim_static as ptorch_optim_static

# ── XOR Dataset ──────────────────────────────────
def make_xor_truth_table(num_bits=2):
    n = 1 << num_bits
    rows = torch.arange(n, dtype=torch.long)
    bits = ((rows.unsqueeze(1) >> torch.arange(num_bits - 1, -1, -1)) & 1)
    labels = (bits.sum(dim=1) % 2).long()
    return bits.float(), labels

X_XOR, Y_XOR = make_xor_truth_table(2)

class ProjectionTracer(torch.fx.Tracer):
    def is_leaf_module(self, m, module_qualified_name):
        if isinstance(m, ProjectionModule):
            return True
        return super().is_leaf_module(m, module_qualified_name)

class PropagateCache(torch.fx.Interpreter):
    def __init__(self, module, skip_modules=None):
        super().__init__(module)
        self.skip_targets = skip_modules or set()

    def run_node(self, n):
        cache_to_apply = None
        if n.op == 'call_module' and n.target not in self.skip_targets:
            submod = self.module.get_submodule(n.target)
            if hasattr(submod, 'projection_forward_cache') and isinstance(submod.projection_forward_cache, list):
                if submod.projection_forward_cache:
                    cache = submod.projection_forward_cache[0]
                    if cache is not None:
                        cache_to_apply = cache
        result = super().run_node(n)
        if cache_to_apply is not None:
            result.data.copy_(cache_to_apply.data)
        return result

class MLP(tnn.Module):
    def __init__(self, hidden=[4], in_features=2, classes=1):
        super().__init__()
        last = in_features
        self.hidden_layers = tnn.ModuleList()
        for i, f in enumerate(hidden):
            if i == 0:
                self.hidden_layers.append(LinearFrozen(last, f, bias=False, g=1.0))
            else:
                self.hidden_layers.append(Linear(last, f, bias=False, g=1.0, alpha=1.0, num_iters=10))
            self.hidden_layers.append(LeakyReLU(0.1))
            last = f
        self.out = Linear(last, 1, bias=False, g=1.0, alpha=1.0, num_iters=10)
        self.loss = HardMarginLoss()

    def forward(self, x, y):
        for i in range(0, len(self.hidden_layers), 2):
            x = self.hidden_layers[i](x)
            x = self.hidden_layers[i+1](x)
        x = self.out(x)
        return self.loss(x, y.view(-1, 1).float())

    def forward_eval(self, x):
        for i in range(0, len(self.hidden_layers), 2):
            x = self.hidden_layers[i](x)
            x = self.hidden_layers[i+1](x)
        x = self.out(x)
        return x

# ─── 1. Print FX graph nodes ───────────────────────────────────────────────
torch.manual_seed(42)
model = MLP()
tracer = ProjectionTracer()
graph = tracer.trace(model)
traced = torch.fx.GraphModule(model, graph)

print("=== FX Graph Nodes ===")
for node in traced.graph.nodes:
    print(f"  op={node.op:15s}  target={str(node.target):30s}  name={node.name}")

# ─── 2. Check which modules have projection_forward_cache ─────────────────
print("\n=== Modules with projection_forward_cache ===")
for name, mod in traced.named_modules():
    if hasattr(mod, 'projection_forward_cache'):
        print(f"  {name!r:30s}  cache={mod.projection_forward_cache}")

# ─── 3. Run a few training steps and verify accuracy ──────────────────────
x_train, y_train = X_XOR, Y_XOR
optimizer = ptorch_optim_static.AlternatingProjectionsMomentum(model.parameters())

traced.train()
output = traced(x_train, y_train)

def eval_acc():
    with torch.no_grad():
        logits = model.forward_eval(x_train)
        preds = (logits > 0.5).long().squeeze()
        return (preds == y_train).float().mean().item()

print(f"\n=== Training (skip_modules={{'out'}}) ===")
print(f"Step 0: acc={eval_acc():.3f}")

for step in range(1, 501):
    optimizer.zero_grad()
    output.sum().backward()
    optimizer.step()

    # Check what's in the caches right after backward
    if step == 1:
        print(f"\nAfter step 1 backward:")
        for name, mod in traced.named_modules():
            if hasattr(mod, 'projection_forward_cache') and mod.projection_forward_cache:
                c = mod.projection_forward_cache[0]
                status = f"shape={c.shape} sum={c.sum().item():.4f}" if c is not None else "None"
                print(f"  {name!r}: {status}")

    interpreter = PropagateCache(traced, skip_modules={'out'})
    output = interpreter.run(x_train, y_train)

    if step % 100 == 0:
        acc = eval_acc()
        print(f"Step {step:4d}: acc={acc:.3f}")

print(f"\nFinal acc: {eval_acc():.3f}")
print("Done.")
