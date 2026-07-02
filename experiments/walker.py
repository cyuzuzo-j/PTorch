"""Projection-based policy gradient for continuous control (Walker2d).

A continuous-action counterpart to pg_ptorch.py, built on the ptorch cyclic
projection framework. The policy is a diagonal Gaussian:

    a ~ N(mu(s), sigma),   sigma = exp(log_std)

  * mu(s) is produced by a projection-aware MLP (Linear + ReLU).
  * log_std is a single learnable leaf parameter, shared across the batch, and
    trained by *its own projection target* (not a raw gradient) — see
    GaussianPolicyGradientProjection in ptorch.core.ops.

Variance reduction uses a projection-aware value baseline V(s) trained toward
discounted reward-to-go (MSEProjection), with the standardized advantage
adv = reward_to_go - V(s) weighting the policy step.

Every projection op's forward returns a pseudo-loss (for logging); its backward
returns a *target* for its output, which the projection layers and
ProjectionAdam consume (pseudo-grad g = p - p_target).

Requires MuJoCo:  pip install "gymnasium[mujoco]"
"""
import math
import torch
import torch.nn as nn
import numpy as np
try:
    import gymnasium as gym          # maintained drop-in, supports NumPy 2.0
except ImportError:
    import gym                       # fall back to legacy gym
Discrete, Box = gym.spaces.Discrete, gym.spaces.Box
import matplotlib
matplotlib.use('Agg')  # headless-safe backend (no display on the cluster)
import matplotlib.pyplot as plt

import ptorch  # noqa: F401  (applies the projection overrides on import)
from ptorch.nn.modules import Linear, ReLU
from ptorch.optim_static import ProjectionMuon, ProjectionAdam
from ptorch.core.ops import GaussianPolicyGradientProjection, MSEProjection


def is_muon_param(p):
    # Muon orthogonalizes 2D weight *matrices*. Biases are stored (1, out) and
    # log_std is 1D — orthogonalizing those is meaningless (and torch.optim.Muon
    # rejects non-2D params outright), so they go to Adam. This is the standard
    # Muon recipe: Muon for hidden matrices, Adam for everything else.
    return p.ndim == 2 and min(p.shape) > 1


def make_optimizers(params, lr, adam_lr, optim='muon'):
    """Build optimizer(s) for a param group.

    optim='adam': a single ProjectionAdam over all params.
    optim='muon': Muon for 2D weight matrices, Adam for biases/scalars (the
                  standard hybrid Muon recipe).
    """
    params = list(params)
    if optim == 'adam':
        return [ProjectionAdam(params, lr=lr)]
    muon_p = [p for p in params if is_muon_param(p)]
    adam_p = [p for p in params if not is_muon_param(p)]
    opts = []
    if muon_p:
        opts.append(ProjectionMuon(muon_p, lr=lr, momentum=0.95, nesterov=True))
    if adam_p:
        opts.append(ProjectionAdam(adam_p, lr=adam_lr))
    return opts


def mlp(sizes, norm='l2'):
    # Build a feedforward projection network (Linear + ReLU, no head activation).
    layers = []
    for j in range(len(sizes)-1):
        layers.append(Linear(sizes[j], sizes[j+1], norm=norm))
        if j < len(sizes)-2:
            layers.append(ReLU(norm=norm))
    return nn.Sequential(*layers)


class RunningMeanStd:
    """Welford running mean/variance for observation normalization.

    Walker2d observations are unbounded; whitening them is close to essential
    for the projection layers (whose targets are parameter-scaled) to learn.
    """
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean += delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta**2 * self.count * batch_count / tot) / tot
        self.count = tot

    def normalize(self, x):
        return (np.asarray(x, dtype=np.float32) - self.mean) / np.sqrt(self.var + 1e-8)


def compute_gae(rews, vals, last_val, gamma, lam):
    """Generalized Advantage Estimation for one episode.

    rews, vals: per-step reward and V(s_t). last_val: bootstrap V(s_T) (0 if the
    episode terminated, V(final next obs) if it was truncated by the time limit).
    Returns (adv, ret) where ret = adv + vals is the value-fit target (TD(lambda)
    return). GAE is the key variance reducer that makes vanilla PG learn on
    MuJoCo; rtg - V (lam=1, no value bootstrap) was too noisy.
    """
    n = len(rews)
    adv = np.zeros(n, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(n)):
        next_v = last_val if t == n - 1 else vals[t + 1]
        delta = rews[t] + gamma * next_v - vals[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    ret = adv + np.asarray(vals, dtype=np.float32)
    return adv, ret


def train(env_name='Walker2d-v5', hidden_sizes=[1024], lr=1e-2, value_lr=1e-2,
          adam_lr=1e-2, optim='muon', epochs=200, batch_size=5000, gamma=0.99,
          lam=0.95, lmbda=0.1, value_iters=25, norm='l2', normalize_obs=True,
          render=False):
    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Box), \
        "walker.py is for continuous (Box) action spaces; use pg_ptorch.py for discrete."

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_low = torch.as_tensor(env.action_space.low, dtype=torch.float32)
    act_high = torch.as_tensor(env.action_space.high, dtype=torch.float32)

    rms = RunningMeanStd(obs_dim) if normalize_obs else None

    def prep(obs):
        # numpy obs -> normalized float32 tensor
        if rms is not None:
            obs = rms.normalize(obs)
        return torch.as_tensor(obs, dtype=torch.float32)

    # ── policy: diagonal Gaussian ───────────────────────────────────────────────
    # mu(s) flows through the projection layers; log_std is a shared learnable
    # leaf trained by its own projection target.
    policy_net = mlp([obs_dim] + hidden_sizes + [act_dim], norm=norm)
    log_std = nn.Parameter(torch.full((act_dim,), math.log(0.5)))
    policy_params = list(policy_net.parameters()) + [log_std]

    # value baseline (projection-aware), trained toward reward-to-go via MSE target
    value_net = mlp([obs_dim] + hidden_sizes + [1], norm=norm)

    def get_action(obs_t):
        # sample an action from the current Gaussian policy (no gradient)
        with torch.no_grad():
            mu = policy_net(obs_t)
            a = mu + torch.exp(log_std) * torch.randn_like(mu)
            a = a.numpy()
        # NaN-safe: a diverging mean head can emit NaN/Inf, and clamp(NaN) stays
        # NaN -> MuJoCo "simulation unstable". Sanitize before stepping the env.
        a = np.nan_to_num(a, nan=0.0, posinf=float(act_high[0]), neginf=float(act_low[0]))
        return np.clip(a, act_low.numpy(), act_high.numpy())

    def compute_loss(obs, act, adv, rtg):
        # policy pseudo-loss -> target for (mu, log_std) in backward
        mu = policy_net(obs)
        policy_loss = GaussianPolicyGradientProjection.apply(
            mu, log_std, act, adv, 5, lmbda)
        # value pseudo-loss (MSE target (v + rtg)/2 -> half-step toward the return)
        v = value_net(obs).squeeze(-1)
        value_loss = MSEProjection.apply(v, rtg)
        return policy_loss + value_loss

    # Muon for the 2D matrices, Adam for biases + log_std (adam_lr scales those).
    policy_opts = make_optimizers(policy_params, lr=lr, adam_lr=adam_lr, optim=optim)
    value_opts = make_optimizers(value_net.parameters(), lr=value_lr, adam_lr=adam_lr, optim=optim)
    all_opts = policy_opts + value_opts

    def train_one_epoch():
        batch_obs = []          # observations (raw, normalized at use time)
        batch_acts = []         # actions
        batch_rtg = []          # discounted reward-to-go (value target)
        batch_rets = []         # full episode returns (logging)
        batch_lens = []         # episode lengths (logging)

        obs, _ = env.reset()
        ep_rews = []
        finished_rendering_this_epoch = False

        while True:
            if (not finished_rendering_this_epoch) and render:
                env.render()

            batch_obs.append(obs.copy())
            act = get_action(prep(obs))
            obs, rew, terminated, truncated, _ = env.step(act)
            done = terminated or truncated

            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)
                batch_rtg += discounted_rtg(ep_rews, gamma).tolist()

                obs, _ = env.reset()
                ep_rews = []
                finished_rendering_this_epoch = True
                if len(batch_obs) > batch_size:
                    break

        # update the obs normalizer on this batch before we form the tensors
        if rms is not None:
            rms.update(np.asarray(batch_obs, dtype=np.float32))

        obs_t = prep(np.asarray(batch_obs, dtype=np.float32))
        rtg = torch.as_tensor(batch_rtg, dtype=torch.float32)

        # advantage = reward-to-go - V(s), standardized. V is detached: it is a
        # weight for the policy step, not part of the value target.
        with torch.no_grad():
            v_pred = value_net(obs_t).squeeze(-1)
        adv = rtg - v_pred
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        acts = torch.as_tensor(np.asarray(batch_acts), dtype=torch.float32)

        for opt in all_opts:
            opt.zero_grad()
        batch_loss = compute_loss(obs=obs_t, act=acts, adv=adv, rtg=rtg)
        batch_loss.backward()
        for opt in all_opts:
            opt.step()
        return batch_loss, batch_rets, batch_lens

    # training loop
    hist_loss, hist_ret, hist_len = [], [], []
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        hist_loss.append(batch_loss.item())
        hist_ret.append(float(np.mean(batch_rets)))
        hist_len.append(float(np.mean(batch_lens)))
        with torch.no_grad():
            ls_mean = log_std.mean().item()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f \t log_std: %.3f' %
              (i, batch_loss.item(), np.mean(batch_rets), np.mean(batch_lens), ls_mean))

    # save a visual summary of training
    epochs_axis = np.arange(len(hist_ret))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs_axis, hist_ret, color='tab:blue')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('average return')
    axes[0].set_title('Return (%s, ptorch norm=%s)' % (env_name, norm))
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_axis, hist_loss, color='tab:red')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('pseudo-loss')
    axes[1].set_title('Policy-gradient + value pseudo-loss')
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = 'walker_training_%s.png' % env_name.replace('/', '-')
    fig.savefig(out_path, dpi=120)
    print('\nSaved training plot to %s' % out_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='Walker2d-v5')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--value_lr', type=float, default=1e-2)
    parser.add_argument('--adam_lr', type=float, default=1e-2,
                        help='Adam lr for biases + log_std (Muon handles 2D matrices)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=5000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lmbda', type=float, default=0.1,
                        help='proximal/ascent step size for the Gaussian policy target')
    parser.add_argument('--hidden', type=int, nargs='+', default=[1024])
    parser.add_argument('--optim', type=str, default='muon', choices=['muon', 'adam'])
    parser.add_argument('--norm', type=str, default='l2', choices=['l2', 'linf'])
    parser.add_argument('--no_normalize_obs', dest='normalize_obs', action='store_false')
    parser.set_defaults(normalize_obs=True)
    args = parser.parse_args()
    print('\nProjection policy gradient (ptorch) for continuous control, with value baseline.\n')
    train(env_name=args.env_name, hidden_sizes=args.hidden, lr=args.lr,
          value_lr=args.value_lr, adam_lr=args.adam_lr, optim=args.optim,
          epochs=args.epochs, batch_size=args.batch_size, gamma=args.gamma,
          lmbda=args.lmbda, norm=args.norm, normalize_obs=args.normalize_obs,
          render=args.render)
