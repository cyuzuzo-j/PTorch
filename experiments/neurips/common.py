"""Shared plumbing for the NeurIPS experiments.

Every experiment is one YAML file, passed to every script with `--config`.
Paths inside a YAML (`results_dir`, `figures_dir`, `data_dir`) are relative
to that YAML file.

Data protocol (identical for every framework):
  train / val : seeded 90% / 10% split of the official training set
  test        : the official test set, evaluated once per run, on the
                checkpoint with the best validation accuracy
Validation drives early stopping and checkpoint selection; the test set is
never used for either. Every CSV row carries `split` ("val" or "test").
"""
import argparse
import glob
import math
import os
import time

import pandas as pd
import tqdm
import yaml


# ── Config ──────────────────────────────────────────────────────────────────

def parse_args(description, variants=False):
    """--config (required) plus a few run filters for smoke tests. All
    experiment settings come from the YAML, never from the command line."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True, help="experiment YAML")
    p.add_argument("--fresh", action="store_true",
                   help="delete each result CSV before this process first writes to it")
    p.add_argument("--tasks", help="comma-separated task names (default: all)")
    if variants:
        p.add_argument("--variants", help="comma-separated variant names (default: all)")
    p.add_argument("--max-steps", type=int, help="smoke test: cap max_steps (0 = no training steps)")
    p.add_argument("--num-runs", type=int, help="smoke test: cap num_runs")
    return p.parse_args()


def load_config(path, args=None):
    """Read a YAML, resolve its paths relative to the file, apply run filters."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    here = os.path.dirname(os.path.abspath(path))
    for key in ("results_dir", "memory_results_dir", "figures_dir", "data_dir"):
        if key in cfg:
            cfg[key] = os.path.normpath(os.path.join(here, cfg[key]))
    if args is not None:
        for key in ("max_steps", "num_runs"):
            if getattr(args, key) is not None:
                cfg[key] = getattr(args, key)
        if args.tasks:
            cfg["tasks"] = select(cfg["tasks"], args.tasks)
    return cfg


def select(entries, names):
    """Keep the entries whose `name` is in the comma-separated `names`."""
    wanted = set(names.split(","))
    return [e for e in entries if e["name"] in wanted]


def seeds(cfg):
    """Per-run seeds: random_seed + r for r = 0 .. num_runs-1."""
    return [cfg["random_seed"] + r for r in range(cfg["num_runs"])]


def free_memory():
    """Release the previous run's tensors so runs don't inherit each other's memory."""
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_data(cfg, task, batch_size, seed, label_noise=0.0):
    """Train/val/test splits for one run. The split itself is seeded by `seed`."""
    from experiments.shared.data import InfiniteCifarDataModule, MNISTDataModule
    datasets = {"MNIST": MNISTDataModule, "CIFAR10": InfiniteCifarDataModule}
    ds = datasets[task["dataset"]](batch_size=batch_size, seed=seed, label_noise=label_noise,
                                   num_classes=task["classes"],
                                   data_dir=cfg.get("data_dir", "./dataset"))
    return ds.train_iterator(), ds.val_dataloader(), ds.test_dataloader()


# ── Training loop ───────────────────────────────────────────────────────────

def sync_time(device=None):
    """Wall-clock timestamp after draining queued CUDA work."""
    import torch
    if device is not None and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


class Timer:
    """Training-only clock: data loading and evaluation are excluded."""
    def __init__(self, device=None):
        self.device = device
        self.elapsed = 0.0
        self._t0 = None

    def resume(self):
        self._t0 = sync_time(self.device)

    def pause(self):
        self.elapsed += sync_time(self.device) - self._t0


def _all_optimizers(optimizer):
    yield optimizer
    nested = getattr(optimizer, "vector_optimizer", None)  # ProjectionMuon's Adam part
    if nested is not None:
        yield nested


def compile_step(train_step, model, optimizer, batch):
    """torch.compile the optimizer step (PTorch). Three warm-up steps on `batch`
    take the compile out of the timed steps; the weights and optimizer state
    are then rewound (fresh state is all zeros for Adam and Muon). Skipped
    under TORCHDYNAMO_DISABLE=1 (the eager column)."""
    import torch
    if os.environ.get("TORCHDYNAMO_DISABLE", "0") == "1":
        return train_step
    optimizer.step = torch.compile(optimizer.step)  # traces into nested optimizers too
    weights = {k: t.detach().clone() for k, t in model.state_dict().items()}
    for _ in range(3):
        train_step(batch)
    with torch.no_grad():
        for k, t in model.state_dict().items():
            t.copy_(weights[k])
        for o in _all_optimizers(optimizer):
            for state in o.state.values():
                for t in state.values():
                    if torch.is_tensor(t):
                        t.zero_()
    sync_time(batch[0].device)
    return train_step


def tensor_lr(kwargs, device):
    """Optimizer kwargs with `lr` as a 0-dim tensor on `device`, for lr_on_regression:
    the compiled optimizer step reads the tensor, so scaling it in place (see
    lr_regression_hooks) never recompiles; a Python-float lr is a dynamo guard and
    would recompile on every change. On CUDA, Adam takes a tensor lr only with
    capturable=True (state steps on the device; same update math)."""
    import torch
    kwargs = dict(kwargs)
    kwargs["lr"] = torch.tensor(float(kwargs.get("lr", 1e-3)), dtype=torch.float32, device=device)
    if torch.device(device).type == "cuda":
        kwargs.setdefault("capturable", True)
    return kwargs


def lr_regression_hooks(cfg, model, optimizer):
    """(on_best, on_regression) callbacks for fit's `lr_on_regression`, or (None, None)
    when the YAML does not set it (so no optimizer-state copies are kept).

    on_best() copies the optimizer state at each new best; on_regression(best, factor)
    copies the best weights and that optimizer state back in place (CUDA graphs keep
    their addresses), then scales every param group's lr by `factor` and returns the
    new lr. State that did not exist at the best snapshot (e.g. a step-0 best in eager
    mode, before Adam's lazy init) is zeroed, which is Adam's and Muon's fresh state."""
    import torch
    if not cfg.get("lr_on_regression"):
        return None, None
    opts, saved = list(_all_optimizers(optimizer)), {}

    def on_best():
        saved.clear()
        for i, o in enumerate(opts):
            for j, st in enumerate(o.state.values()):
                for k, t in st.items():
                    if torch.is_tensor(t):
                        saved[i, j, k] = t.clone()

    @torch.no_grad()
    def on_regression(best, factor):
        for k, t in model.state_dict().items():
            t.copy_(best[k])
        lr = None
        for i, o in enumerate(opts):
            for j, st in enumerate(o.state.values()):
                for k, t in st.items():
                    if torch.is_tensor(t):
                        t.copy_(saved[i, j, k]) if (i, j, k) in saved else t.zero_()
            for g in o.param_groups:
                if torch.is_tensor(g["lr"]):
                    g["lr"].mul_(factor)
                else:
                    g["lr"] *= factor
                lr = float(g["lr"])
        return lr

    return on_best, on_regression


def set_precision(cfg):
    """`precision: highest` turns TF32 off for matmuls and cuDNN convs (strict fp32,
    so results do not depend on the GPU generation); unset keeps PyTorch's defaults."""
    import torch
    if cfg.get("precision") == "highest":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")


def add_schedule_meta(cfg, meta):
    """Record the YAML's lr_on_regression schedule (if any) in a run's CSV metadata."""
    if cfg.get("lr_on_regression"):
        meta["lr_on_regression"] = "drop={drop} factor={factor} min_lr={min_lr}".format(**cfg["lr_on_regression"])


def accuracy(predict, loader):
    """Sample-weighted accuracy. `predict(x, y)` returns (n_correct, n) for one batch."""
    correct = total = 0
    for x, y in loader:
        c, n = predict(x, y)
        correct += float(c)
        total += int(n)
    return correct / max(total, 1)


def fit(cfg, next_batch, train_step, val_acc, snapshot, timer, desc, max_steps=None,
        on_best=None, on_regression=None):
    """Train with validation every `eval_every` steps (null: only at the end) and early stopping
    after `patience` evaluations without improvement.

    next_batch()      -> batch (not timed)
    train_step(batch) -> None  (timed)
    val_acc()         -> validation accuracy
    snapshot()        -> copy of the current weights

    Stops at `max_steps` or, if the YAML sets `time_budget_s`, after that many
    seconds of training. Returns (val curve, steps taken, best-val snapshot).

    If the YAML sets `lr_on_regression: {drop, factor, min_lr}`, an evaluation
    more than `drop` below the best so far calls on_regression(best, factor),
    which rewinds to the best snapshot, scales the lr by `factor` and returns
    the new lr; training stops once that lr is below `min_lr`. on_best() runs
    after each new best snapshot (e.g. to copy the optimizer state).
    """
    max_steps = next((s for s in (max_steps, cfg.get("max_steps")) if s is not None), math.inf)
    budget = cfg.get("time_budget_s", math.inf)
    curve, best_acc, best, stale, step = [], -1.0, None, 0, 0
    with tqdm.tqdm(total=None if max_steps == math.inf else max_steps, desc=desc, unit="step") as bar:
        while True:
            done = step >= max_steps or timer.elapsed >= budget
            if done or (cfg["eval_every"] and step % cfg["eval_every"] == 0):
                acc = val_acc()
                curve.append((step, acc, timer.elapsed))
                if acc > best_acc:
                    best_acc, best, stale = acc, snapshot(), 0
                    if on_best is not None:
                        on_best()
                else:
                    stale += 1
                bar.set_postfix(val=f"{acc:.4f}", best=f"{best_acc:.4f}")
                sched = cfg.get("lr_on_regression")
                if sched and on_regression is not None and not done and acc < best_acc - sched["drop"]:
                    lr = on_regression(best, sched["factor"])
                    tqdm.tqdm.write(f"LR step {step}: val {acc:.4f} < best {best_acc:.4f} - {sched['drop']}"
                                    f" -> rewind to best, lr {lr:.3g}")
                    if lr < sched["min_lr"]:
                        tqdm.tqdm.write(f"LR below min_lr {sched['min_lr']:.3g}, stopping at step {step}")
                        break
                if stale >= cfg["patience"]:
                    print(f"Early stop at step {step}")
                    break
            if done:
                break
            batch = next_batch()
            timer.resume()
            train_step(batch)
            timer.pause()
            step += 1
            bar.update(1)
    return curve, step, best


# ── Results ─────────────────────────────────────────────────────────────────

_CLEARED = set()


def save_run(cfg, meta, curve, steps, test_acc, timer, fresh=False, **test_extra):
    """Append one run to <results_dir>/<variant>_<task>.csv: one `val` row per
    evaluation and a final `test` row carrying the timing / memory columns."""
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    rows = [{**meta, "split": "val", "step": s, "acc": a, "wall_time_s": w, "timestamp": stamp}
            for s, a, w in curve]
    rows.append({**meta, "split": "test", "step": steps, "acc": test_acc,
                 "wall_time_s": timer.elapsed, "ms_per_step": 1e3 * timer.elapsed / max(steps, 1),
                 "timestamp": stamp, **test_extra})
    path = os.path.join(cfg["results_dir"], f"{meta['variant']}_{meta['task']}.csv")
    if fresh and path not in _CLEARED and os.path.exists(path):
        os.remove(path)
    _CLEARED.add(path)
    os.makedirs(cfg["results_dir"], exist_ok=True)
    pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    print(f"Test acc {test_acc:.4f}, train time {timer.elapsed:.1f}s -> {path}")


def load_runs(cfg, split="val"):
    """All result CSVs of an experiment, one split, with a display `Label` column from the
    YAML's `labels:` map (`<name>_eager` defaults to `<name>`'s + " (eager)", else the variant)."""
    paths = sorted(glob.glob(os.path.join(cfg["results_dir"], "*.csv")))
    if not paths:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if "split" not in df.columns:
        raise ValueError(f"{cfg['results_dir']} has CSVs without a split column; rerun with --fresh")
    df = df[df["split"] == split].copy()
    eager = {f"{k}_eager": f"{v} (eager)" for k, v in cfg.get("labels", {}).items()}
    df["Label"] = df["variant"].map({**eager, **cfg.get("labels", {})}).fillna(df["variant"])
    return df
