"""
Equivalence test for the TD(λ) trace: td_lambda = 0.0 must be byte-for-byte
identical to the pre-trace code.

Builds a baseline copy of src/ptorch in a tempdir with the three trace
additions surgically removed from core/ops.py, then trains the same
fixed-seed depth-4 identity MLP for 20 steps in three subprocesses:

  1. baseline code (PYTHONPATH shadow)            -> hash_base
  2. current code, td_lambda = 0.0 (default)      -> hash_zero
  3. current code, td_lambda = 0.5                -> hash_half

Asserts hash_base == hash_zero (exact no-op at λ=0) and hash_zero != hash_half
(the mechanism actually does something).

Run:  TORCHDYNAMO_DISABLE=1 python experiments/deep/test_td_lambda_equiv.py
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO_SRC = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
)


def child(lam, td_mode="vector"):
    import hashlib
    import random

    import torch

    import ptorch
    from ptorch.config import config
    from ptorch.nn.modules import Linear
    from ptorch.optim_static import ProjectionMuonV2
    from ptorch.core.ops import MSEProjection

    if lam != 0.0:
        config.update("td_lambda", lam)
        config.update("td_mode", td_mode)

    torch.manual_seed(0)
    random.seed(0)
    layers = torch.nn.ModuleList([Linear(32, 32, bias=True) for _ in range(4)])
    opt = ProjectionMuonV2(layers.parameters())

    loss = None
    for _ in range(20):
        x = torch.randn(16, 32)
        y = x.clone()
        opt.zero_grad()
        h = x
        for layer in layers:
            h = layer(h)
        loss = MSEProjection.apply(h, y)
        loss.backward()
        opt.step()

    hasher = hashlib.sha256()
    for p in layers.parameters():
        hasher.update(p.detach().cpu().contiguous().numpy().tobytes())
    print("PTORCH", ptorch.__file__)
    print("HASH", hasher.hexdigest(), "LOSS", f"{loss.item():.17g}")


def build_baseline(tmpdir):
    """Copy src/ptorch and strip the three TD-trace additions from ops.py."""
    dst = os.path.join(tmpdir, "ptorch")
    shutil.copytree(os.path.join(REPO_SRC, "ptorch"), dst,
                    ignore=shutil.ignore_patterns("__pycache__"))
    ops_path = os.path.join(dst, "core", "ops.py")
    with open(ops_path) as f:
        text = f.read()

    subs = [
        # 1. TD_TRACE global (comment block + dict) after DIAG = None
        (r"\n\n# TD\(λ\) eligibility trace over depth.*?"
         r"TD_TRACE = \{\"g\": None, \"seed_norm\": None, \"floor\": None\}\n",
         "\n"),
        # 2. seed block in MSEProjection.backward
        (r"\n        if config\.td_lambda != 0\.0:\n"
         r"            seed = \(predictions - out\)\.detach\(\)\n"
         r"            TD_TRACE\[\"g\"\] = seed\n"
         r"            TD_TRACE\[\"seed_norm\"\] = seed\.norm\(\)\n"
         r"            TD_TRACE\[\"floor\"\] = float\(TD_TRACE\[\"seed_norm\"\]\)\n",
         "\n"),
        # 1b. repeater block -> identity passthroughs (call sites and the
        # modules_experimental import stay valid)
        (r"# ── TD direction-pipe repeater \(non-bilinear nodes\) ─+\n"
         r".*?"
         r"# ── end TD repeater ─+\n",
         "def td_repeat_target(x_det, x_bar):\n"
         "    return x_bar\n\n\n"
         "def process_node_target(x_det, x_bar):\n"
         "    return process_activation_target(x_det, x_bar)\n"),
        # 2b. seed block in CrossEntropyProjection.backward
        (r"\n        if config\.td_lambda != 0\.0:\n"
         r"            seed = \(logits - x\)\.detach\(\)\n"
         r"            TD_TRACE\[\"g\"\] = seed\n"
         r"            TD_TRACE\[\"seed_norm\"\] = seed\.norm\(\)\n"
         r"            TD_TRACE\[\"floor\"\] = float\(TD_TRACE\[\"seed_norm\"\]\)\n",
         "\n"),
        # 3. blend block in MatMulProjection.backward -> original return
        (r"\n        A_ret = process_activation_target\(A_det, A_proj\)\n"
         r".*?return A_ret, B_proj, None, None, None, None, None, None, None",
         "\n        return process_activation_target(A_det, A_proj), "
         "B_proj, None, None, None, None, None, None, None"),
    ]
    for pattern, repl in subs:
        text, n = re.subn(pattern, repl, text, count=1, flags=re.DOTALL)
        assert n == 1, f"baseline surgery failed for pattern: {pattern[:60]}..."
    assert "TD_TRACE" not in text, "baseline ops.py still references TD_TRACE"

    with open(ops_path, "w") as f:
        f.write(text)
    return tmpdir


def run_child(lam, pythonpath=None, td_mode="vector"):
    env = dict(os.environ)
    env["TORCHDYNAMO_DISABLE"] = "1"
    if pythonpath is not None:
        env["PYTHONPATH"] = pythonpath
    else:
        env.pop("PYTHONPATH", None)
    out = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--child",
         "--lam", str(lam), "--td-mode", td_mode],
        env=env, capture_output=True, text=True, check=True,
    ).stdout
    ptorch_file = re.search(r"^PTORCH (.+)$", out, re.M).group(1)
    hash_ = re.search(r"^HASH (\w+) LOSS (.+)$", out, re.M)
    return ptorch_file, hash_.group(1), hash_.group(2)


def main():
    tmpdir = tempfile.mkdtemp(prefix="ptorch_td_baseline_")
    try:
        build_baseline(tmpdir)

        base_file, hash_base, loss_base = run_child(0.0, pythonpath=tmpdir)
        assert base_file.startswith(tmpdir), \
            f"baseline child imported {base_file}, not the shadow copy"
        cur_file, hash_zero, loss_zero = run_child(0.0)
        assert cur_file.startswith(REPO_SRC), \
            f"current child imported {cur_file}, not the repo"
        _, hash_half, loss_half = run_child(0.5)
        _, hash_zero_n, loss_zero_n = run_child(0.0, td_mode="norm")
        _, hash_norm, loss_norm = run_child(0.9, td_mode="norm")

        print(f"baseline (pre-trace code):        {hash_base}  loss={loss_base}")
        print(f"current, lam=0.0:                 {hash_zero}  loss={loss_zero}")
        print(f"current, lam=0.5, mode=vector:    {hash_half}  loss={loss_half}")
        print(f"current, lam=0.0, mode=norm:      {hash_zero_n}  loss={loss_zero_n}")
        print(f"current, lam=0.9, mode=norm:      {hash_norm}  loss={loss_norm}")

        assert hash_base == hash_zero and loss_base == loss_zero, \
            "FAIL: td_lambda=0.0 is NOT byte-for-byte identical to baseline"
        assert hash_base == hash_zero_n, \
            "FAIL: td_lambda=0.0 with mode=norm is NOT identical to baseline"
        assert hash_zero != hash_half, \
            "FAIL: vector mode lam=0.5 produced identical weights (trace inactive?)"
        assert hash_zero != hash_norm, \
            "FAIL: norm mode lam=0.9 produced identical weights (floor inactive?)"
        print("PASS: td_lambda=0.0 is byte-for-byte identical to the pre-trace "
              "code in both modes, and both modes change the dynamics when on.")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--lam", type=float, default=0.0)
    parser.add_argument("--td-mode", type=str, default="vector")
    args = parser.parse_args()
    if args.child:
        child(args.lam, args.td_mode)
    else:
        main()
