#!/usr/bin/env bash
# Run all layer profiling scripts and collect results.
set -e
cd "$(dirname "$0")"

for script in Linear.py Embedding.py ReLU.py Conv2D.py FftConv2D.py MaxPool2D.py MultiHeadAttention.py; do
    echo "=========================================="
    echo "  Profiling: $script"
    echo "=========================================="
    python "$script"
    echo ""
done

echo "All profiling complete."
echo "Outputs saved under experiments/profiling/outputs/"
echo "  *_graph.png   — computation graph visualizations"
echo "  *_trace/      — JAX profiler traces (open with Perfetto)"
echo "  *_memory.prof — device memory profiles"
