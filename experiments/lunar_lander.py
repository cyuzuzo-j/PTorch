"""Projection-based policy gradient for continuous control (LunarLander).

A deliberately minimal continuous-action policy gradient built on the ptorch
cyclic-projection framework. The policy is a diagonal Gaussian

    a ~ N(mu(s), sigma),   sigma = exp(log_std)

  * mu(s) comes from a projection-aware MLP (Linear + Tanh).
  * log_std is a single learnable leaf trained by its own projection target
    (GaussianPolicyGradientProjection in ptorch.core.ops).

Variance reduction is a projection-aware value baseline V(s) trained toward the
discounted reward-to-go (MSEProjection); the standardized advantage
adv = reward_to_go - V(s) weights the policy step.

Each projection op's forward returns a pseudo-loss (logging only); its backward
returns a *target* for its output, which the projection layers and ProjectionAdam
consume as a pseudo-grad g = p - p_target.

LunarLanderContinuous-v3 (Box2D): 8-dim observation, 2 continuous engine actions
in [-1, 1], reward for a soft fuel-efficient landing (solving threshold ~200).

Requires the Box2D backend:  pip install "gymnasium[box2d]"
"""
import os
import math
import torch
import torch.nn as nn
import numpy as np
try:
    import gymnasium as gym
except ImportError:
    import gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import ptorch  # noqa: F401  (applies the projection overrides on import)
from ptorch.nn.modules import Linear, ReLU
from ptorch.optim_static import ProjectionAdam
from ptorch.core.ops import GaussianPolicyGradientProjection, MSEProjection


def mlp(sizes, norm='l2'):
    # feedforward projection network (Linear + ReLU, no head activation)
    layers = []
    for j in range(len(sizes) - 1):
        layers.append(Linear(sizes[j], sizes[j + 1], norm=norm))
        if j < len(sizes) - 2:
            layers.append(ReLU(norm=norm))
    return nn.Sequential(*layers)


class RunningMeanStd:
    """Welford running mean/variance for observation normalization."""
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        bmean, bvar, bcount = x.mean(0), x.var(0), x.shape[0]
        delta = bmean - self.mean
        tot = self.count + bcount
        self.mean += delta * bcount / tot
        self.var = (self.var * self.count + bvar * bcount
                    + delta ** 2 * self.count * bcount / tot) / tot
        self.count = tot

    def normalize(self, x):
        return (np.asarray(x, dtype=np.float32) - self.mean) / np.sqrt(self.var + 1e-8)


def compute_gae(rews, vals, last_val, gamma, lam):
    # GAE(lambda) for one episode. delta_t = r_t + gamma V(s_{t+1}) - V(s_t),
    # A_t = delta_t + gamma lam A_{t+1}. last_val bootstraps V(s_T): 0 on a real
    # terminal, V(next_obs) on a time-limit truncation. Returns advantages and
    # value targets (returns = advantage + V). Far lower variance than raw
    # reward-to-go, which is what lets the policy take useful steps.
    T = len(rews)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_v = last_val if t == T - 1 else vals[t + 1]
        delta = rews[t] + gamma * next_v - vals[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    return adv, adv + vals


def train(env_name='LunarLanderContinuous-v3', hidden_sizes=(64, 64),
          lr=3e-4, value_lr=1e-3, epochs=200, batch_size=4000, gamma=0.99,
          gae_lambda=0.95, lmbda=0.05, update_epochs=10, minibatch_size=512,
          adv_clip=3.0, target_kl=0.015, value_warmup=10,
          log_std_init=-0.5, log_std_final=-1.3,
          max_delta=0.3, norm='l2', seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_low = torch.as_tensor(env.action_space.low, dtype=torch.float32)
    act_high = torch.as_tensor(env.action_space.high, dtype=torch.float32)

    rms = RunningMeanStd(obs_dim)      # observation whitening
    ret_std = None                     # running std of returns (reward scaling)

    def prep(obs):
        return torch.as_tensor(rms.normalize(obs), dtype=torch.float32)

    policy_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], norm=norm)
    value_net = mlp([obs_dim] + list(hidden_sizes) + [1], norm=norm)
    # log_std is a fixed (annealed) exploration schedule, NOT learned: training it
    # through the Gaussian projection op collapses sigma to the floor, which
    # detonates the 1/sigma^2 mu-target. A buffer (not a Parameter) keeps it out
    # of the optimizer; the op's target for it is simply discarded.
    log_std = torch.full((1, act_dim), float(log_std_init))

    policy_opt = ProjectionAdam(policy_net.parameters(), lr=lr)
    value_opt = ProjectionAdam(value_net.parameters(), lr=value_lr)

    def get_action(obs_t):
        with torch.no_grad():
            mu = policy_net(obs_t)
            a = mu + torch.exp(log_std) * torch.randn_like(mu)
            a = torch.clamp(a, act_low, act_high).numpy().reshape(-1)
            return np.nan_to_num(a, nan=0.0)

    def eval_deterministic(n=10, max_steps=1000):
        rets = []
        for _ in range(n):
            obs, _ = env.reset()
            done, steps, ret = False, 0, 0.0
            with torch.no_grad():
                while not done and steps < max_steps:
                    mu = policy_net(prep(obs))
                    a = torch.clamp(mu, act_low, act_high).numpy().reshape(-1)
                    obs, r, term, trunc, _ = env.step(np.nan_to_num(a, nan=0.0))
                    ret += r
                    done = term or trunc
                    steps += 1
            rets.append(ret)
        return float(np.mean(rets))

    def collect(update_scale=True):
        # Phase 1: roll out raw episodes. Phase 2: set the reward scale from this
        # batch, then compute GAE in scaled-reward space (rewards / ret_std) so the
        # value targets land ~O(1). The scale is only updated during value warmup
        # (policy frozen -> stationary returns) and then frozen, so the value net
        # fits a *stationary* target instead of chasing a moving reward scale.
        nonlocal ret_std
        episodes = []          # (ep_obs, ep_acts, ep_rews, boot_obs or None)
        ep_rets, ep_lens = [], []
        obs, _ = env.reset()
        ep_obs, ep_acts, ep_rews = [], [], []
        n = 0
        while n < batch_size:
            a = get_action(prep(obs))
            nobs, r, term, trunc, _ = env.step(a)
            ep_obs.append(obs.copy()); ep_acts.append(a); ep_rews.append(r)
            obs = nobs; n += 1
            if term or trunc:
                episodes.append((ep_obs, ep_acts, ep_rews, None if term else obs.copy()))
                ep_rets.append(float(np.sum(ep_rews))); ep_lens.append(len(ep_rews))
                obs, _ = env.reset()
                ep_obs, ep_acts, ep_rews = [], [], []

        if ret_std is None or update_scale:
            batch_std = max(float(np.std(ep_rets)), 1.0)
            ret_std = batch_std if ret_std is None else 0.9 * ret_std + 0.1 * batch_std

        obs_buf, act_buf, adv_buf, ret_buf = [], [], [], []
        for ep_obs, ep_acts, ep_rews, boot in episodes:
            with torch.no_grad():
                v = value_net(prep(np.asarray(ep_obs, dtype=np.float32))).squeeze(-1).numpy()
                last_v = 0.0 if boot is None else float(value_net(prep(boot)).item())
            r = np.asarray(ep_rews, dtype=np.float32) / ret_std
            a, ret = compute_gae(r, v, last_v, gamma, gae_lambda)
            obs_buf.extend(ep_obs); act_buf.extend(ep_acts)
            adv_buf.extend(a); ret_buf.extend(ret)

        obs_np = np.asarray(obs_buf, dtype=np.float32)
        rms.update(obs_np)
        return (obs_np, np.asarray(act_buf, dtype=np.float32),
                np.asarray(adv_buf, dtype=np.float32),
                np.asarray(ret_buf, dtype=np.float32), ep_rets, ep_lens)

    def update(obs_np, acts_np, adv_np, rets_np, freeze_policy=False):
        obs_t = prep(obs_np)
        acts = torch.as_tensor(acts_np)
        ret_t = torch.as_tensor(rets_np)
        adv = torch.as_tensor(adv_np)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv = adv.clamp(-adv_clip, adv_clip)
        # collection-policy mean, fixed for this batch -> trust-region reference
        with torch.no_grad():
            mu_old = policy_net(obs_t)
        var2 = 2.0 * torch.exp(2.0 * log_std).mean()      # 2 sigma^2

        N = len(obs_np)
        idx = np.arange(N)
        pol_acc = val_acc = nb = 0
        kl = 0.0
        stop = False
        for _ in range(update_epochs):
            if stop:
                break
            np.random.shuffle(idx)
            for s in range(0, N, minibatch_size):
                mb = torch.as_tensor(idx[s:s + minibatch_size])
                # trust region: freeze the policy once its mean has drifted too
                # far from collection (fixed-sigma Gaussian KL = ||mu-mu_old||^2
                # / 2sigma^2). Without it the off-policy steps run mu out of
                # range and the proximal target detonates. Value keeps fitting.
                with torch.no_grad():
                    kl = float((((policy_net(obs_t) - mu_old) ** 2).sum(-1) / var2).mean())
                stop = kl > target_kl
                policy_opt.zero_grad(); value_opt.zero_grad()
                mu = policy_net(obs_t[mb])
                pol = GaussianPolicyGradientProjection.apply(
                    mu, log_std, acts[mb], adv[mb], 1, lmbda, max_delta)
                v = value_net(obs_t[mb]).squeeze(-1)
                val = MSEProjection.apply(v, ret_t[mb])
                (pol + val).backward()
                # value warmup: hold the policy frozen for the first few epochs so
                # a fresh (random) value net fits before its noisy advantages are
                # allowed to move (and wreck) the policy.
                if not stop and not freeze_policy:
                    policy_opt.step()
                value_opt.step()
                pol_acc += float(pol.detach()); val_acc += float(val.detach()); nb += 1
        return pol_acc / nb, val_acc / nb, kl

    import copy
    hist_ret, hist_det = [], []
    best_det = -1e9
    best_state = None
    for i in range(epochs):
        # linear exploration anneal: start wide, commit to a low-noise landing
        frac = i / float(max(epochs - 1, 1))
        log_std.fill_(log_std_init + (log_std_final - log_std_init) * frac)
        warm = i < value_warmup
        obs_np, acts_np, adv_np, rets_np, ep_rets, ep_lens = collect(update_scale=warm)
        pol_loss, val_loss, kl = update(obs_np, acts_np, adv_np, rets_np,
                                        freeze_policy=warm)
        mean_ret = float(np.mean(ep_rets))
        hist_ret.append(mean_ret)
        det = ''
        if i % 5 == 0 or i == epochs - 1:
            # deterministic (greedy mu) eval is the real metric; advantages are
            # standardized each batch, so once near-optimal the policy random-walks
            # on noise -> snapshot the best so post-peak drift can't lose it.
            det_ret = eval_deterministic()
            hist_det.append((i, det_ret))
            if det_ret > best_det:
                best_det = det_ret
                best_state = (copy.deepcopy(policy_net.state_dict()),
                              log_std.clone(), copy.deepcopy(rms.__dict__))
                det = '  <-- best'
            det = '\t det_ret: %8.2f%s' % (det_ret, det)
        print('epoch %3d | pol %8.2f | val %7.2f | kl %.3f | return %8.2f | ep_len %6.1f | log_std %.2f%s'
              % (i, pol_loss, val_loss, kl, mean_ret, float(np.mean(ep_lens)),
                 log_std.mean().item(), det))

    # restore + save the best-by-det_ret policy (guards against post-peak drift)
    if best_state is not None:
        policy_net.load_state_dict(best_state[0])
        log_std.copy_(best_state[1])
        rms.__dict__.update(best_state[2])
        torch.save({'policy': best_state[0], 'log_std': best_state[1],
                    'rms': best_state[2], 'det_ret': best_det}, 'lunar_lander_best.pt')

    # training plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist_ret, color='tab:blue', alpha=0.6, label='stochastic return')
    if hist_det:
        xs, ys = zip(*hist_det)
        ax.plot(xs, ys, color='tab:red', marker='o', ms=3, label='deterministic return')
    ax.axhline(200, color='gray', ls='--', lw=0.8, label='solved (200)')
    ax.set_xlabel('epoch'); ax.set_ylabel('return')
    ax.set_title('%s (ptorch projection PG, norm=%s)' % (env_name, norm))
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = 'lunar_lander_training_%s.png' % env_name.replace('/', '-')
    fig.savefig(out, dpi=120)
    print('\nbest det_ret %.2f  ->  saved %s' % (best_det, out))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--env_name', '--env', default='LunarLanderContinuous-v3')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=4000)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--value_lr', type=float, default=1e-3)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--gae_lambda', type=float, default=0.95)
    p.add_argument('--lmbda', type=float, default=0.05)
    p.add_argument('--update_epochs', type=int, default=10)
    p.add_argument('--minibatch_size', type=int, default=512)
    p.add_argument('--adv_clip', type=float, default=3.0)
    p.add_argument('--target_kl', type=float, default=0.015)
    p.add_argument('--value_warmup', type=int, default=10)
    p.add_argument('--log_std_init', type=float, default=-0.5)
    p.add_argument('--log_std_final', type=float, default=-1.3)
    p.add_argument('--max_delta', type=float, default=0.3)
    p.add_argument('--hidden', type=int, nargs='+', default=[64, 64])
    p.add_argument('--norm', default='l2', choices=['l2', 'linf'])
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    train(env_name=a.env_name, hidden_sizes=a.hidden, lr=a.lr, value_lr=a.value_lr,
          epochs=a.epochs, batch_size=a.batch_size, gamma=a.gamma,
          gae_lambda=a.gae_lambda, lmbda=a.lmbda,
          update_epochs=a.update_epochs, minibatch_size=a.minibatch_size,
          adv_clip=a.adv_clip, target_kl=a.target_kl, value_warmup=a.value_warmup,
          log_std_init=a.log_std_init, log_std_final=a.log_std_final,
          max_delta=(a.max_delta if a.max_delta > 0 else None),
          norm=a.norm, seed=a.seed)
