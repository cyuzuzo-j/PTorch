#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "Running deep MLP (4L) benchmark with ptorch (sweeps over muon_activations)..."
python bench_ptorch.py --config deep_mlp_config.yaml

echo "Running deep MLP (4L) benchmark with torch (reference)..."
python bench_torch.py --config deep_mlp_config.yaml

echo "Done! Results saved to results/deep_mlp"
