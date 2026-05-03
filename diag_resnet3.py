"""Diagnostic 3: trace feature maps through each layer to find collapse point."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'frameworks')))

import torch
import torch.nn.functional as F
import ptorch.config as ptorch_config
from ptorch.core.overrides import apply_overrides

apply_overrides()
ptorch_config.use_projections = True

from experiments.cnn_benchmarks.bench_ptorch_resnet import ResNet8_PTorch, ProjectedGlobalAvgPool
from experiments.shared.data import InfiniteCifarDataModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
model = ResNet8_PTorch(classes=10, in_channels=3).to(device)

ds = InfiniteCifarDataModule(batch_size=8, data_dir="./dataset", seed=42)
train_iter = ds.train_iterator()
x_np, y_np = next(train_iter)
x = torch.tensor(x_np, dtype=torch.float32, device=device)
if x.shape[-1] in [1, 3]:
    x = x.permute(0, 3, 1, 2)

print(f"Input: shape={x.shape}, std={x.std():.4f}, mean={x.mean():.4f}")

with torch.no_grad():
    # init_conv + relu
    out = model.init_conv(x)
    print(f"After init_conv: shape={out.shape}, std={out.std():.6f}, mean={out.mean():.6f}, range=[{out.min():.6f}, {out.max():.6f}]")

    out = model.init_relu(out)
    print(f"After init_relu: shape={out.shape}, std={out.std():.6f}, mean={out.mean():.6f}, range=[{out.min():.6f}, {out.max():.6f}]")

    # block1
    b1_in = out
    conv_in = b1_in  # Would be detached in forward, but for feature check we skip that
    b1_out = model.block1.relu1(model.block1.conv1(conv_in))
    print(f"Block1 conv1+relu: shape={b1_out.shape}, std={b1_out.std():.6f}")
    b1_out = model.block1.conv2(b1_out)
    print(f"Block1 conv2 (zero-init): shape={b1_out.shape}, std={b1_out.std():.6f}, max={b1_out.abs().max():.6f}")
    # skip path
    if model.block1.shortcut is not None:
        b1_skip = model.block1.shortcut(b1_in)
        print(f"Block1 shortcut: shape={b1_skip.shape}, std={b1_skip.std():.6f}")
    else:
        b1_skip = b1_in
        print(f"Block1 skip (identity): shape={b1_skip.shape}, std={b1_skip.std():.6f}")
    b1_sum = b1_skip + b1_out
    print(f"Block1 skip+out: std={b1_sum.std():.6f}")
    out = model.block1.relu2(b1_sum)
    print(f"After block1: shape={out.shape}, std={out.std():.6f}")

    # block2
    b2_in = out
    conv_in = b2_in
    b2_out = model.block2.relu1(model.block2.conv1(conv_in))
    print(f"Block2 conv1+relu: shape={b2_out.shape}, std={b2_out.std():.6f}")
    b2_out = model.block2.conv2(b2_out)
    print(f"Block2 conv2 (zero-init): shape={b2_out.shape}, std={b2_out.std():.6f}, max={b2_out.abs().max():.6f}")
    b2_skip = model.block2.shortcut(b2_in)
    print(f"Block2 shortcut: shape={b2_skip.shape}, std={b2_skip.std():.6f}")
    b2_sum = b2_skip + b2_out
    out = model.block2.relu2(b2_sum)
    print(f"After block2: shape={out.shape}, std={out.std():.6f}")

    # block3
    b3_in = out
    conv_in = b3_in
    b3_out = model.block3.relu1(model.block3.conv1(conv_in))
    print(f"Block3 conv1+relu: shape={b3_out.shape}, std={b3_out.std():.6f}")
    b3_out = model.block3.conv2(b3_out)
    print(f"Block3 conv2 (zero-init): shape={b3_out.shape}, std={b3_out.std():.6f}, max={b3_out.abs().max():.6f}")
    b3_skip = model.block3.shortcut(b3_in)
    print(f"Block3 shortcut: shape={b3_skip.shape}, std={b3_skip.std():.6f}")
    b3_sum = b3_skip + b3_out
    out = model.block3.relu2(b3_sum)
    print(f"After block3: shape={out.shape}, std={out.std():.6f}")

    # global avg pool
    pooled = ProjectedGlobalAvgPool.apply(out)
    print(f"After GAP: shape={pooled.shape}, std={pooled.std():.6f}")

    # Check per-sample variance
    per_sample_std = pooled.std(dim=0).mean()
    print(f"Per-sample variation (std across batch dim, mean over features): {per_sample_std:.6f}")

    # head
    logits = model.head(pooled)
    print(f"Logits: shape={logits.shape}, std={logits.std():.6f}, per-sample std={logits.std(dim=0).mean():.6f}")
    print(f"Sample logits[0]: {logits[0]}")
    print(f"Sample logits[1]: {logits[1]}")
