# Experiment Guidelines

This document outlines how experiments should be structured and executed in this repository. It is intended to be read by both humans and AI agents.

## Core Principles

1. **Single Purpose:** Each experiment must do specifically **one thing**. Do not mix multiple independent concepts into a single experiment. If you need to test something else, create a new experiment folder.
2. **Framework Parity:** Every experiment must contain implementations (versions) for **all three frameworks**:
   - `pjax`
   - `pjax_orr`
   - `ptorch`
   Ensure that the experimental setup, data loading, and tracking are consistent across these framework implementations to allow for fair comparisons. Whenever possible, **a single combined file should be created for both `pjax` and `pjax_orr`** to maximize code reuse.
3. **Tracking:** Use **Weights & Biases (`wandb`)** for all experiment tracking. Do not use other tracking systems (e.g., Aim is deprecated).
   - **Clear Naming:** Each experiment and run should be easily trackable. Use clear, descriptive names for your `wandb` runs so they can be easily identified and filtered later.
   - **Comprehensive Logging:** Ensure all relevant metrics, configurations, and **all relevant hyperparameters** are properly logged to `wandb` to ensure reproducibility.
4. **Reusability & Brevity:** Prioritize code reusability and brevity. **Do not recreate the same files everywhere again.**
   - Put common code, such as data loaders, metrics, and plotting utilities, in the `experiments/shared/` directory.
   - Import and reuse these shared utilities in your framework-specific experiment scripts.
   - Keep experiment scripts concise, focusing only on the framework-specific initialization, the training loop, and the `wandb` logging logic.

## Experiment Structure Example

A typical experiment folder (e.g., `experiments/my_new_experiment/`) should look like this:

```
my_new_experiment/
├── run_pjax_combined.py # Unified pjax and pjax_orr implementation using wandb
├── run_ptorch.py        # ptorch implementation using wandb
└── README.md            # Optional: Brief description of what this specefic experiment tests
```

Shared resources should be placed in `experiments/shared/` and imported into your `run_*.py` scripts.

## Running Experiments

When asked to create or run an experiment, adhere strictly to the guidelines above. Ensure that the `wandb` initialization is correct and the run finishes appropriately.
