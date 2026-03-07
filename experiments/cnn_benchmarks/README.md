# CNN Benchmarks

This directory contains benchmarking scripts for evaluating a shared CNN proxy-net architecture across multiple frameworks:
- `pjax` (Native JAX proxy-net implementation)
- `pjax_orr` (Original JAX proxy-net implementation)
- `ptorch` (PyTorch with proxy-net extensions)
- `torch` (Standard PyTorch baseline)

## Configuration
The benchmarks are driven entirely by `config.yaml`.
You can adjust:
- Number of runs
- Max steps
- Supported datasets/tasks (`MNIST`, `CIFAR10`)
- Batch sizes
- Optimizer configurations for each framework

## Running Benchmarks

### PJAX
Run the standard PJAX implementation or the original PJAX-ORR implementation across all tasks and batch sizes defined in `config.yaml`.

```bash
python experiments/cnn_benchmarks/bench_pjax_combined.py --impl pjax
python experiments/cnn_benchmarks/bench_pjax_combined.py --impl pjax_orr
```

### PTorch
Run the PTorch framework implementation.

```bash
python experiments/cnn_benchmarks/bench_ptorch.py
```

### Standard Torch Base
Run the standard PyTorch baseline directly.

```bash
python experiments/cnn_benchmarks/bench_torch.py
```

## Logging
All scripts automatically log to `wandb` using the `experiment_name` specified in `config.yaml`. The framework, task, epoch, accuracy, and all relevant optimizer hyper-parameters are recorded dynamically.
