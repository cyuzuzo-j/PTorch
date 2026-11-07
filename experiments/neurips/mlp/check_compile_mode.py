"""PTORCH_COMPILE_MODE check: the ablation MLP (3072-1024-10, B=256,
ProjectionMuon, proximal CE) trained for a few steps on random data, once in
torch's default compile mode and once in PTORCH_COMPILE_MODE (default
reduce-overhead), same seed and batches. Reports the max relative parameter
difference, host (CPU) ms/step of each, and any "skipping cudagraphs".

    PTORCH_COMPILE_DYNAMIC=0 PTORCH_COMPILE_MODE=reduce-overhead PYTHONPATH=src:. \
        python -m experiments.neurips.mlp.check_compile_mode [--steps 50]

Each mode runs in its own subprocess (ops.py reads the env var at import).
On CPU it still runs: reduce-overhead compiles but skips cudagraphs, so only
the parameter match means anything there.
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
TASK = {"name": "CIFAR10_1024", "in_features": 3072, "classes": 10, "hidden": [1024]}


def child(args):
    import torch
    import torch.nn.functional as F
    from torch._dynamo.utils import counters
    import torch._inductor.cudagraph_utils as cg_utils
    import ptorch.nn.modules as pnn
    import ptorch.optim_static as poptim
    from ptorch.core import ops
    from experiments.neurips.mlp.bench_ptorch import MLP  # also sets matmul precision

    skips = []

    class Grab(logging.Handler):
        def emit(self, record):
            skips.append(record.getMessage())

    cg_utils.cudagraphs_log.addHandler(Grab(logging.WARNING))
    cg_utils.cudagraphs_log.setLevel(logging.WARNING)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    mark_step = torch.compiler.cudagraph_mark_step_begin if ops._TC_MODE else (lambda: None)

    torch.manual_seed(args.seed)
    model = MLP(TASK, "l2", None, "fused", None).to(device)
    loss = pnn.CrossEntropy()
    opt = poptim.ProjectionMuon(model.parameters(), lr=1e-3, weight_decay=0.1)
    gen = torch.Generator().manual_seed(args.seed + 1)
    batches = [(torch.randn(args.batch, TASK["in_features"], generator=gen).to(device),
                F.one_hot(torch.randint(0, TASK["classes"], (args.batch,), generator=gen),
                          TASK["classes"]).float().to(device)) for _ in range(8)]

    host = []
    sync()
    t_wall = None
    for i in range(args.steps):
        if i == args.warmup:
            sync()
            t_wall = time.perf_counter()
        x, y = batches[i % len(batches)]
        t0 = time.perf_counter()
        mark_step()
        loss(model(x), y).sum().backward()
        opt.step()
        opt.zero_grad(set_to_none=False)
        host.append(time.perf_counter() - t0)
    sync()
    timed = args.steps - args.warmup
    wall = (time.perf_counter() - t_wall) / timed if t_wall is not None and timed > 0 else float("nan")
    torch.save({k: t.detach().cpu() for k, t in model.state_dict().items()}, args.child + ".pt")
    with open(args.child + ".json", "w") as f:
        json.dump({"mode": ops._TC_MODE, "dynamic": ops._TC_DYNAMIC, "device": str(device),
                   "host_ms": 1e3 * sum(host[args.warmup:]) / max(timed, 1),
                   "wall_ms": 1e3 * wall,
                   "cudagraph_skips": counters["inductor"]["cudagraph_skips"],
                   "skip_msgs": sorted(set(skips))}, f)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10, help="untimed leading steps (compile, record)")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--child", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.child:
        return child(args)

    import torch
    base = dict(os.environ)
    base.setdefault("PTORCH_COMPILE_DYNAMIC", "0")
    mode = base.pop("PTORCH_COMPILE_MODE", None) or "reduce-overhead"
    res = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, env in [("default", base), (mode, {**base, "PTORCH_COMPILE_MODE": mode})]:
            out = os.path.join(tmp, name)
            cmd = [sys.executable, "-m", "experiments.neurips.mlp.check_compile_mode",
                   "--child", out, "--steps", str(args.steps), "--warmup", str(args.warmup),
                   "--batch", str(args.batch), "--seed", str(args.seed)]
            subprocess.run(cmd, cwd=ROOT, env=env, check=True)
            with open(out + ".json") as f:
                res[name] = json.load(f)
            res[name]["params"] = torch.load(out + ".pt")

    ref, new = res["default"]["params"], res[mode]["params"]
    rel = {k: float((new[k] - ref[k]).norm() / ref[k].norm().clamp_min(1e-30)) for k in ref}
    worst = max(rel, key=rel.get)
    print(f"\ndevice {res['default']['device']}, PTORCH_COMPILE_DYNAMIC={base['PTORCH_COMPILE_DYNAMIC']}, "
          f"{args.steps} steps (first {args.warmup} untimed)")
    for name in ("default", mode):
        r = res[name]
        print(f"  {name:>16}: host {r['host_ms']:.3f} ms/step, wall {r['wall_ms']:.3f} ms/step, "
              f"cudagraph_skips {r['cudagraph_skips']}")
        for m in r["skip_msgs"]:
            print(f"    {m}")
    print(f"  max relative param diff: {rel[worst]:.3e} ({worst})")
    print(f"  max abs param diff: {max(float((new[k] - ref[k]).abs().max()) for k in ref):.3e}")


if __name__ == "__main__":
    main()
