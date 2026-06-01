##################################################
###   Synthetic sorting benchmark                ###
###   Input: 5 random numbers                    ###
###   Target: rank of each input position (0..4) ###
###   Loss: per-position cross-entropy           ###
###                                              ###
###   Compares Sort vs ReLU as the activation    ###
###   in a small MLP.                            ###
##################################################
import sys, os, time, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../frameworks')))
import torch
import torch.nn as tnn
import torch.nn.functional as F
import pandas as pd
import tqdm

from ptorch.nn.modules import Linear, ReLU, Sort, CrossEntropy
from ptorch import config as ptorch_config
import ptorch.optim_static as ptorch_optim_static

ptorch_config.use_projections = True
ptorch_config.config.use_muon_activations = False

N_ITEMS = 5
HIDDEN = 64
BATCH = 512
STEPS = 4000
EVAL_EVERY = 100
EVAL_BATCH = 4096

ACTIVATIONS = {
    "ReLU": lambda: ReLU(),
    "Sort": lambda: Sort(),
}


def make_batch(batch_size, n=N_ITEMS, device='cpu'):
    """5 random numbers per sample; target = rank of each position (argsort^-1)."""
    x = torch.randn(batch_size, n, device=device)
    _, indices = torch.sort(x, dim=-1)        # x[indices[k]] is the k-th smallest
    ranks = torch.argsort(indices, dim=-1)    # ranks[i] = rank of x[i]
    return x, ranks


class MLP(tnn.Module):
    def __init__(self, act_name, hidden=HIDDEN, n=N_ITEMS, depth=2):
        super().__init__()
        act = ACTIVATIONS[act_name]
        layers = [Linear(n, hidden), act()]
        for _ in range(depth - 1):
            layers += [Linear(hidden, hidden), act()]
        layers.append(Linear(hidden, n * n))
        self.net = tnn.Sequential(*layers)
        self.n = n

    def forward(self, x):
        return self.net(x).reshape(-1, self.n, self.n)


_ce_module = CrossEntropy()

def per_pos_ce(logits, ranks):
    """logits: (B, N, N) — N positions × N rank classes. ranks: (B, N) class IDs.

    Routes through the projection-aware CrossEntropy so the target propagated
    back to the projection Linears is a logits target, not a vanilla gradient.
    """
    B, N, _ = logits.shape
    flat_logits = logits.reshape(B * N, N)
    flat_ranks = ranks.reshape(B * N)
    ranks_oh = F.one_hot(flat_ranks, num_classes=N).float()
    return _ce_module(flat_logits, ranks_oh)


def eval_metrics(model, device, n_batches=4):
    model.eval()
    pos_correct, total_pos, exact_match, total_samples = 0, 0, 0, 0
    with torch.no_grad():
        for _ in range(n_batches):
            x, ranks = make_batch(EVAL_BATCH, device=device)
            logits = model(x)
            pred = logits.argmax(dim=-1)         # (B, N)
            pos_correct += (pred == ranks).sum().item()
            total_pos += ranks.numel()
            exact_match += (pred == ranks).all(dim=-1).sum().item()
            total_samples += ranks.size(0)
    model.train()
    return pos_correct / total_pos, exact_match / total_samples


def run_one(act_name, seed, device, opt_name='ProjectionAdam', lr=1e-3):
    torch.manual_seed(seed)
    model = MLP(act_name).to(device)
    OptCls = getattr(ptorch_optim_static, opt_name)
    opt = OptCls(model.parameters(), lr=lr)

    rows = []
    t0 = time.time()
    pbar = tqdm.tqdm(range(STEPS), desc=f"{act_name} seed={seed}", unit="step")
    for step in pbar:
        x, ranks = make_batch(BATCH, device=device)
        logits = model(x)
        loss = per_pos_ce(logits, ranks)
        loss.backward()
        opt.step(); opt.zero_grad()

        if step % EVAL_EVERY == 0 or step == STEPS - 1:
            pos_acc, exact = eval_metrics(model, device)
            rows.append({
                'activation': act_name, 'seed': seed, 'step': step,
                'loss': float(loss.detach()), 'pos_acc': pos_acc,
                'exact_match': exact, 'elapsed_s': time.time() - t0,
            })
            pbar.set_postfix(loss=f"{float(loss.detach()):.4f}",
                             pos=f"{pos_acc:.3f}", exact=f"{exact:.3f}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", nargs="*", default=list(ACTIVATIONS.keys()))
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    parser.add_argument("--opt", default="ProjectionAdam")
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device}, projections={ptorch_config.use_projections}")

    all_rows = []
    for act in args.activations:
        for seed in args.seeds:
            print(f"\n=== {act} seed={seed} ===")
            all_rows.extend(run_one(act, seed, device, opt_name=args.opt, lr=args.lr))

    out_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f'argsort_n{N_ITEMS}_{args.opt}.csv')
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    # Summary: best exact match per activation (over seeds, over training).
    df = pd.DataFrame(all_rows)
    print("\nBest exact-match per (activation, seed):")
    summary = df.groupby(['activation', 'seed'])['exact_match'].max().unstack(fill_value=0.0)
    print(summary.to_string())
    print("\nMean best exact-match per activation:")
    print(df.groupby(['activation', 'seed'])['exact_match'].max().groupby('activation').agg(['mean', 'std']).to_string())
