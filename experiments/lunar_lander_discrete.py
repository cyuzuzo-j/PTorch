"""Projection-based policy gradient for discrete control (LunarLander-v3).

The discrete counterpart to lunar_lander.py. Because the policy is a softmax over
4 actions, the projection step uses the *bounded* discrete op

    logits <- logits + lambda * w * (onehot(a) - softmax(logits))      (iterated)

(PolicyGradientProjection in ptorch.core.ops). Unlike the continuous Gaussian op
— whose mu-target carries a 1/sigma^2 factor that detonates as sigma shrinks —
onehot - softmax lives in (-1, 1), so the policy target can never run away. That
makes this far more stable than the continuous version, which is why discrete
LunarLander is the better showcase of the projection framework on RL.

Same variance-reduction machinery as lunar_lander.py:
  * value baseline V(s) trained toward the GAE return via MSEProjection,
  * GAE(lambda) advantages weighting the policy step,
  * reward scaling so the value targets land ~O(1),
  * value warmup (fit V before it moves the policy),
  * a categorical-KL trust region (stop once the policy drifts from collection),
  * best-by-deterministic-return snapshotting.

Every projection op's forward returns a pseudo-loss (logging only); its backward
returns a *target* the projection layers and ProjectionAdam consume as g = p - p_target.

LunarLander-v3 (Box2D): 8-dim observation, 4 discrete actions (no-op, fire
left/main/right engine), shaped reward, solving threshold ~200.

Requires the Box2D backend:  pip install "gymnasium[box2d]"
"""
import os
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
from ptorch.core.ops import PolicyGradientProjection, MSEProjection


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
    # value targets (returns = advantage + V).
    T = len(rews)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_v = last_val if t == T - 1 else vals[t + 1]
        delta = rews[t] + gamma * next_v - vals[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    return adv, adv + vals


def train(env_name='LunarLander-v3', hidden_sizes=(64, 64),
          lr=5e-3, value_lr=5e-3, epochs=200, batch_size=5000, gamma=0.99,
          gae_lambda=0.95, lmbda=1.0, proj_steps=3, update_epochs=10,
          minibatch_size=512, adv_clip=3.0, target_kl=0.02, value_warmup=10,
          norm='l2', seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    rms = RunningMeanStd(obs_dim)      # observation whitening
    ret_std = None                     # running std of returns (reward scaling)

    def prep(obs):
        return torch.as_tensor(rms.normalize(obs), dtype=torch.float32)

    logits_net = mlp([obs_dim] + list(hidden_sizes) + [n_acts], norm=norm)
    value_net = mlp([obs_dim] + list(hidden_sizes) + [1], norm=norm)
    policy_opt = ProjectionAdam(logits_net.parameters(), lr=lr)
    value_opt = ProjectionAdam(value_net.parameters(), lr=value_lr)

    def get_action(obs_t):
        with torch.no_grad():
            probs = torch.softmax(logits_net(obs_t), dim=-1)
            return int(torch.multinomial(probs, 1).item())

    def eval_deterministic(n=10, max_steps=1000):
        rets = []
        for _ in range(n):
            obs, _ = env.reset()
            done, steps, ret = False, 0, 0.0
            with torch.no_grad():
                while not done and steps < max_steps:
                    a = int(logits_net(prep(obs)).argmax(-1).item())
                    obs, r, term, trunc, _ = env.step(a)
                    ret += r
                    done = term or trunc
                    steps += 1
            rets.append(ret)
        return float(np.mean(rets))

    def collect(update_scale=True):
        # roll out raw episodes, set the reward scale (only during warmup, then
        # frozen, so value targets stay stationary), then compute GAE in
        # scaled-reward space so value targets land ~O(1).
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
        return (obs_np, np.asarray(act_buf, dtype=np.int64),
                np.asarray(adv_buf, dtype=np.float32),
                np.asarray(ret_buf, dtype=np.float32), ep_rets, ep_lens)

    def update(obs_np, acts_np, adv_np, rets_np, freeze_policy=False):
        obs_t = prep(obs_np)
        acts = torch.as_tensor(acts_np)
        ret_t = torch.as_tensor(rets_np)
        adv = torch.as_tensor(adv_np)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv = adv.clamp(-adv_clip, adv_clip)
        # collection-policy log-probs, fixed for this batch -> KL reference
        with torch.no_grad():
            logp_old = torch.log_softmax(logits_net(obs_t), dim=-1)

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
                # trust region: freeze the policy once it has drifted too far from
                # collection (categorical KL(old||new) = sum p_old (logp_old -
                # logp_new)). Value keeps fitting. The discrete op is bounded so
                # this is a soft guard, not a divergence stopgap.
                with torch.no_grad():
                    logp_new = torch.log_softmax(logits_net(obs_t), dim=-1)
                    kl = float((logp_old.exp() * (logp_old - logp_new)).sum(-1).mean())
                stop = kl > target_kl
                policy_opt.zero_grad(); value_opt.zero_grad()
                logits = logits_net(obs_t[mb])
                pol = PolicyGradientProjection.apply(logits, acts[mb], adv[mb], proj_steps, lmbda)
                v = value_net(obs_t[mb]).squeeze(-1)
                val = MSEProjection.apply(v, ret_t[mb])
                (pol + val).backward()
                # value warmup: hold the policy frozen for the first few epochs so
                # a fresh value net fits before its noisy advantages move the policy.
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
        warm = i < value_warmup
        obs_np, acts_np, adv_np, rets_np, ep_rets, ep_lens = collect(update_scale=warm)
        pol_loss, val_loss, kl = update(obs_np, acts_np, adv_np, rets_np, freeze_policy=warm)
        mean_ret = float(np.mean(ep_rets))
        hist_ret.append(mean_ret)
        det = ''
        if i % 5 == 0 or i == epochs - 1:
            det_ret = eval_deterministic()
            hist_det.append((i, det_ret))
            if det_ret > best_det:
                best_det = det_ret
                best_state = (copy.deepcopy(logits_net.state_dict()),
                              copy.deepcopy(rms.__dict__))
                det = '  <-- best'
            det = '\t det_ret: %8.2f%s' % (det_ret, det)
        print('epoch %3d | pol %7.2f | val %7.2f | kl %.3f | return %8.2f | ep_len %6.1f%s'
              % (i, pol_loss, val_loss, kl, mean_ret, float(np.mean(ep_lens)), det))

    # restore + save the best-by-det_ret policy
    if best_state is not None:
        logits_net.load_state_dict(best_state[0])
        rms.__dict__.update(best_state[1])
        torch.save({'policy': best_state[0], 'rms': best_state[1], 'det_ret': best_det},
                   'lunar_lander_discrete_best.pt')

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
    out = 'lunar_lander_discrete_training_%s.png' % env_name.replace('/', '-')
    fig.savefig(out, dpi=120)
    print('\nbest det_ret %.2f  ->  saved %s' % (best_det, out))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--env_name', '--env', default='LunarLander-v3')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=5000)
    p.add_argument('--lr', type=float, default=5e-3)
    p.add_argument('--value_lr', type=float, default=5e-3)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--gae_lambda', type=float, default=0.95)
    p.add_argument('--lmbda', type=float, default=1.0,
                   help='proximal step size inside the discrete projection op')
    p.add_argument('--proj_steps', type=int, default=3,
                   help='inner projection iterations in the op backward')
    p.add_argument('--update_epochs', type=int, default=10)
    p.add_argument('--minibatch_size', type=int, default=512)
    p.add_argument('--adv_clip', type=float, default=3.0)
    p.add_argument('--target_kl', type=float, default=0.02)
    p.add_argument('--value_warmup', type=int, default=10)
    p.add_argument('--hidden', type=int, nargs='+', default=[64, 64])
    p.add_argument('--norm', default='l2', choices=['l2', 'linf'])
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    train(env_name=a.env_name, hidden_sizes=a.hidden, lr=a.lr, value_lr=a.value_lr,
          epochs=a.epochs, batch_size=a.batch_size, gamma=a.gamma,
          gae_lambda=a.gae_lambda, lmbda=a.lmbda, proj_steps=a.proj_steps,
          update_epochs=a.update_epochs, minibatch_size=a.minibatch_size,
          adv_clip=a.adv_clip, target_kl=a.target_kl, value_warmup=a.value_warmup,
          norm=a.norm, seed=a.seed)
