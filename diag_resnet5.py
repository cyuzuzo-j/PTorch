"""Diagnostic 5: compare ResNet vs LeNet, and test muon_activations toggle."""
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

# Import LeNet5 from bench_ptorch copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experiments/cnn_benchmarks'))
# Can't import from "bench_ptorch copy" directly, so import manually
import importlib.util
spec = importlib.util.spec_from_file_location("bench_copy", "experiments/cnn_benchmarks/bench_ptorch copy.py")
bench_copy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench_copy)
LeNet5_PTorch = bench_copy.LeNet5_PTorch

configs = [
    ("ResNet8 + muon_act", ResNet8_PTorch, True),
    ("ResNet8 - muon_act", ResNet8_PTorch, False),
    ("LeNet5 + muon_act", LeNet5_PTorch, True),
    ("LeNet5 - muon_act", LeNet5_PTorch, False),
]

for name, model_cls, use_muon_act in configs:
    print(f"\n=== {name} ===")
    ptorch_config.muon_activations = use_muon_act
    torch.manual_seed(42)
    model = model_cls(classes=10, in_channels=3).to(device)
    ds = InfiniteCifarDataModule(batch_size=128, data_dir="./dataset", seed=42)
    train_iter = ds.train_iterator()
    optimizer = ptorch_optim_static.AlternatingProjections(model.parameters(), lr=1.0)

    for step in range(300):
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

        if step % 50 == 0 or step == 299:
            with torch.no_grad():
                logits = model.forward_eval(x_batch)
            ce_loss = float(F.cross_entropy(logits, y_batch))
            train_acc = float((logits.argmax(dim=-1) == y_batch).float().mean())
            print(f"  step={step:4d}  loss={ce_loss:.4f}  acc={train_acc:.4f}")

ptorch_config.muon_activations = True  # restore default
