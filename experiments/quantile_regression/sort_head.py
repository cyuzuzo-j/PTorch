##################################################
###   Conditional quantile regression demo       ###
###                                              ###
###   Toy: y = sin(x) + (0.1 + |x|) * N(0,1)     ###
###     => closed-form conditional deciles       ###
###                                              ###
###   Compare two heads at the output:           ###
###     Unconstrained: Linear(H, 9)              ###
###     Sort:          Linear(H, 9) → Sort()     ###
###                                              ###
###   Metrics: pinball loss, decile-crossing      ###
###   rate, 80% coverage. Plot predicted deciles. ###
##################################################
import sys, os, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))
import argparse
import torch
import torch.nn as nn
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ptorch.nn.modules import Sort
from ptorch import config as ptorch_config

# Vanilla autograd everywhere; Sort falls through to torch.sort(...).values,
# which has a real backward (scatter through the permutation). The point of
# the demo is the *forward* structural prior — no crossings, by construction.
ptorch_config.use_projections = False

QUANTILES = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
K = QUANTILES.numel()


# ── Data ─────────────────────────────────────────────────────────────────────

def sample(N, device, x=None):
    """y = sin(x) + sigma(x) * eps, eps ~ N(0,1), sigma(x) = 0.1 + |x|."""
    if x is None:
        x = (torch.rand(N, 1, device=device) * 6.0) - 3.0
    sigma = 0.1 + x.abs()
    eps = torch.randn_like(x)
    y = torch.sin(x) + sigma * eps
    return x, y.squeeze(-1)


def true_deciles(x, taus):
    """Closed-form conditional deciles for the heteroscedastic-Gaussian model."""
    sigma = 0.1 + x.abs()                                # (N, 1)
    z = torch.erfinv(2.0 * taus - 1.0) * math.sqrt(2.0)  # (K,) standard normal quantiles
    return torch.sin(x) + sigma * z                      # (N, K)


# ── Models ───────────────────────────────────────────────────────────────────

class _Trunk(nn.Module):
    def __init__(self, in_dim=1, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, K),
        )

    def forward(self, x):
        return self.net(x)


class UnconstrainedHead(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.trunk = _Trunk(hidden=hidden)

    def forward(self, x):
        return self.trunk(x)


class SortHead(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.trunk = _Trunk(hidden=hidden)
        self.sort = Sort()

    def forward(self, x):
        return self.sort(self.trunk(x))


# ── Loss & metrics ───────────────────────────────────────────────────────────

def pinball_loss(pred, y, taus):
    """pred: (B, K), y: (B,), taus: (K,). Mean over batch and quantiles."""
    diff = y.unsqueeze(-1) - pred
    return torch.maximum(taus * diff, (taus - 1.0) * diff).mean()


def crossing_stats(pred):
    """Fraction of samples with any crossing, and mean number of crossings."""
    diffs = pred[:, 1:] - pred[:, :-1]
    any_cross = (diffs < 0).any(dim=-1).float().mean().item()
    mean_cross = (diffs < 0).float().sum(dim=-1).mean().item()
    return any_cross, mean_cross


def coverage_80(pred, y):
    """Fraction of y inside [pred_q10, pred_q90]. Target value: 0.8."""
    lo = pred[:, 0]; hi = pred[:, -1]
    return ((y >= lo) & (y <= hi)).float().mean().item()


# ── Training ─────────────────────────────────────────────────────────────────

def train(model_cls, seed, device, steps=15000, batch=1024, lr=3e-3, log_every=500):
    torch.manual_seed(seed)
    model = model_cls().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    taus = QUANTILES.to(device)

    rows = []
    for step in range(steps):
        x, y = sample(batch, device)
        pred = model(x)
        loss = pinball_loss(pred, y, taus)
        opt.zero_grad(); loss.backward(); opt.step()

        if step % log_every == 0 or step == steps - 1:
            with torch.no_grad():
                xv, yv = sample(8192, device)
                pv = model(xv)
                eval_loss = pinball_loss(pv, yv, taus).item()
                any_cross, mean_cross = crossing_stats(pv)
                cov = coverage_80(pv, yv)
            rows.append({
                'model': model_cls.__name__, 'seed': seed, 'step': step,
                'train_pinball': float(loss.detach()),
                'eval_pinball': eval_loss,
                'crossing_any': any_cross, 'crossing_count': mean_cross,
                'coverage80': cov,
            })
    return model, rows


# ── Plot ─────────────────────────────────────────────────────────────────────

def plot_deciles(models, device, out_path):
    """Side-by-side: predicted deciles vs ground-truth deciles, with raw samples."""
    xv = torch.linspace(-3, 3, 400, device=device).unsqueeze(-1)
    with torch.no_grad():
        true = true_deciles(xv, QUANTILES.to(device)).cpu().numpy()

    # Raw samples for background
    xs, ys = sample(1500, device)
    xs_np, ys_np = xs.cpu().numpy().squeeze(), ys.cpu().numpy()

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, (name, model) in zip(axes, models.items()):
        model.eval()
        with torch.no_grad():
            pred = model(xv).cpu().numpy()
        ax.scatter(xs_np, ys_np, s=4, alpha=0.15, color='gray', label='samples')
        for k in range(K):
            ax.plot(xv.cpu().numpy().squeeze(), true[:, k], '--',
                    color='black', alpha=0.4, linewidth=0.8,
                    label='true' if k == 0 else None)
            ax.plot(xv.cpu().numpy().squeeze(), pred[:, k],
                    label=f'q{int(QUANTILES[k]*100):02d}' if k in (0, 4, 8) else None,
                    linewidth=1.4)
        any_cross, mean_cross = crossing_stats(torch.from_numpy(pred))
        ax.set_title(f'{name}\ncrossings: any={any_cross:.1%}  mean#={mean_cross:.2f}')
        ax.set_xlabel('x')
        ax.legend(fontsize=8, loc='upper left')
    axes[0].set_ylabel('y / deciles')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"Wrote {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=15000)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device}")

    all_rows = []
    final_models = {}  # one per architecture for the plot (seed=0)
    for cls in (UnconstrainedHead, SortHead):
        for seed in args.seeds:
            print(f"\n=== {cls.__name__} seed={seed} ===")
            model, rows = train(cls, seed, device, steps=args.steps)
            all_rows.extend(rows)
            if seed == args.seeds[0]:
                final_models[cls.__name__] = model
            last = rows[-1]
            print(f"  eval pinball={last['eval_pinball']:.4f}  "
                  f"crossings(any/#)={last['crossing_any']:.1%}/{last['crossing_count']:.2f}  "
                  f"coverage80={last['coverage80']:.3f}")

    df = pd.DataFrame(all_rows)
    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    csv_path = os.path.join(out_dir, 'quantile_demo.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    # Final-step summary
    final = df.groupby(['model', 'seed']).tail(1)
    summary = final.groupby('model').agg(
        eval_pinball_mean=('eval_pinball', 'mean'),
        eval_pinball_std=('eval_pinball', 'std'),
        crossing_any_mean=('crossing_any', 'mean'),
        crossing_count_mean=('crossing_count', 'mean'),
        coverage80_mean=('coverage80', 'mean'),
    )
    print("\n" + "=" * 70)
    print("Final-step summary (mean over seeds):")
    print(summary.to_string())
    print(f"\nTarget coverage80 = 0.800")

    plot_deciles(final_models, device, os.path.join(out_dir, 'quantile_demo.png'))
