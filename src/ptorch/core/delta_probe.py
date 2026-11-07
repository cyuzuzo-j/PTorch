"""δ-attenuation instrumentation (§4.2 protocol, per attention block).

Self-contained observation layer for the vanishing-target measurements: at a
probe point holding forward value ``x``, the backward "grad" in this framework
is the *target* ``x̄``, so the target displacement is ``δ = ‖x̄ − x‖₂``. A
probe is a plain ``tensor.register_hook`` that records ``δ`` and returns None
— it inserts **no** autograd node, so the projection chain (and TD floor
routing, which walks grad_fn producers) is structurally untouched. Safe to
attach to legacy and fused paths alike; overlaying the two validates the
"fusion bought memory, not signal" claim.

Everything is inert unless the module-level RECORDER is explicitly configured
(no env var is read at import time), and recording is subsampled by step, so
the default training path is unchanged and un-synced. The ``float(norm)``
GPU→CPU syncs happen only on recorded steps.

Usage (see experiments/attn_projection_native/launch.py):

    from ptorch.core.delta_probe import RECORDER, instrument_blocklists
    RECORDER.configure(path="delta.jsonl", every_n=50)
    instrument_blocklists()          # hooks every nn.ModuleList of Blocks

Intra-attention segment records (post-QK / post-activation / post-PV) are
emitted by CausalSelfAttentionNative and FusedCausalAttentionProjectionV2
(ptorch/{nn,core}/attention_native.py) through the same RECORDER.
"""

from __future__ import annotations

import atexit
import json
import os
from typing import Optional

import torch
import torch.nn as nn


class DeltaRecorder:
    """Append-only JSONL sink for δ scalars, subsampled by training step.

    A "step" here is one training micro-forward (counted by the first hooked
    block, or by RECORDER.step() if a script drives it manually). ``active()``
    is a cheap Python check used to gate every probe, so an unconfigured
    recorder costs one attribute read per probe site.
    """

    def __init__(self):
        self.enabled = False
        self.every_n = 50
        self.path: Optional[str] = None
        self._step = -1  # first step() call lands on 0 -> recorded
        self._rows: list = []
        self._atexit_registered = False

    # ── lifecycle ────────────────────────────────────────────────────────────
    def configure(self, path: str, every_n: int = 50):
        self.path = path
        self.every_n = max(int(every_n), 1)
        self.enabled = True
        self._step = -1
        if not self._atexit_registered:
            atexit.register(self.flush)
            self._atexit_registered = True

    def disable(self):
        self.flush()
        self.enabled = False

    def step(self):
        """Advance the step counter (one training micro-forward)."""
        if self.enabled:
            self._step += 1

    def active(self) -> bool:
        return self.enabled and self._step >= 0 and (self._step % self.every_n == 0)

    # ── recording ────────────────────────────────────────────────────────────
    def record(self, label: str, **scalars):
        if not self.active():
            return
        row = {"step": self._step, "label": label}
        for key, val in scalars.items():
            row[key] = float(val)
        self._rows.append(row)
        if len(self._rows) >= 4096:
            self.flush()

    def flush(self):
        if not self._rows or self.path is None:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "a") as fh:
            for row in self._rows:
                fh.write(json.dumps(row) + "\n")
        self._rows.clear()


RECORDER = DeltaRecorder()


def probe(x: torch.Tensor, label: str) -> torch.Tensor:
    """Record ``δ = ‖x̄ − x‖`` when the target reaches ``x`` in backward.

    Returns ``x`` unchanged (same object — no graph mutation). The hook runs
    once per backward, computes the displacement against the detached forward
    value, and returns None so the target passes through bit-identically.
    """
    if not (RECORDER.active() and torch.is_grad_enabled() and x.requires_grad):
        return x
    x_det = x.detach()

    def _hook(target, _label=label, _x=x_det):
        RECORDER.record(_label,
                        delta=(target.detach() - _x).norm(),
                        x_norm=_x.norm())
        return None

    x.register_hook(_hook)
    return x


def _hook_block(block: nn.Module, label: str, is_first: bool):
    """Forward hooks on one transformer block: δ_in at the block input, δ_out
    at the block output. The first block also drives the step counter (one
    training micro-forward per step)."""

    def _pre(module, inputs):
        if is_first and module.training and torch.is_grad_enabled():
            RECORDER.step()
        if inputs and isinstance(inputs[0], torch.Tensor):
            probe(inputs[0], f"{label}.in")
        return None

    def _post(module, inputs, output):
        if isinstance(output, torch.Tensor):
            probe(output, f"{label}.out")
        return None

    block.register_forward_pre_hook(_pre)
    block.register_forward_hook(_post)


def attach_block_probes(model: nn.Module, block_class_name: str = "Block"):
    """Attach δ_in/δ_out probes to every direct child of every nn.ModuleList
    whose element class is named ``block_class_name`` (the train scripts'
    transformer blocks). Call once, after model construction."""
    idx = 0
    for _, module in model.named_modules():
        if isinstance(module, nn.ModuleList):
            for child in module:
                if type(child).__name__ == block_class_name:
                    _hook_block(child, f"block{idx}", is_first=(idx == 0))
                    idx += 1
    return idx


def instrument_blocklists(block_class_name: str = "Block"):
    """Model-construction interception for scripts we cannot edit: wraps
    ``nn.ModuleList.__init__`` so that any ModuleList built from modules whose
    class is named ``block_class_name`` gets per-block δ probes at creation
    time. Idempotent; instrumented runs only (launch.py)."""
    if getattr(nn.ModuleList, "_ptorch_delta_patched", False):
        return
    orig_init = nn.ModuleList.__init__

    def patched_init(self, modules=None):
        modules = list(modules) if modules is not None else None
        orig_init(self, modules)
        if modules and all(type(m).__name__ == block_class_name for m in modules):
            base = instrument_blocklists._count
            for i, child in enumerate(modules):
                _hook_block(child, f"block{base + i}", is_first=(base + i == 0))
            instrument_blocklists._count += len(modules)

    instrument_blocklists._count = 0
    nn.ModuleList.__init__ = patched_init
    nn.ModuleList._ptorch_delta_patched = True
