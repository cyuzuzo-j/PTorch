"""Projection-based policy gradient for continuous control (BipedalWalker).

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

BipedalWalker-v3 (Box2D): 24-dim observation (hull state + leg joints + 10 lidar
rangefinders), 4 continuous torques in [-1, 1], reward for moving forward minus a
small torque cost and a -100 penalty for falling (solving threshold ~300).

Requires the Box2D backend:  pip install "gymnasium[box2d]"
"""
import os
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
import imageio.v2 as imageio
from PIL import Image

import ptorch  # noqa: F401  (applies the projection overrides on import)
from ptorch.nn.modules import Linear, ReLU
from ptorch.optim_static import ProjectionAdam
from ptorch.core.ops import GaussianPolicyGradientProjection, MSEProjection


def mlp(sizes, norm='linf'):
    # Build a feedforward projection network (Linear + ReLU, no head activation).
    layers = []
    for j in range(len(sizes)-1):
        layers.append(Linear(sizes[j], sizes[j+1], norm=norm))
        if j < len(sizes)-2:
            layers.append(ReLU(norm=norm))
    return nn.Sequential(*layers)


class RunningMeanStd:
    """Welford running mean/variance for observation normalization.

    BipedalWalker mixes bounded lidar readings with unbounded hull/joint
    velocities; whitening them keeps the projection layers (whose targets are
    parameter-scaled) in a sane regime.
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


def gaussian_logp(act, mu, log_std):
    # log N(act; mu, exp(log_std)) summed over the action dimension -> (B,)
    std = torch.exp(log_std)
    return -0.5 * (((act - mu) / std) ** 2 + 2.0 * log_std + math.log(2.0 * math.pi)).sum(-1)


def compute_gae(rews, vals, last_val, gamma, lam):
    # Generalized Advantage Estimation for a single episode.
    #   delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    #   A_t     = delta_t + gamma * lam * A_{t+1}
    # last_val bootstraps V(s_T): 0 if the episode terminated (the walker fell),
    # V(next_obs) if it was merely truncated at the time limit. Returns the
    # advantages and the value targets (returns = advantage + V(s_t)).
    T = len(rews)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_v = last_val if t == T - 1 else vals[t + 1]
        delta = rews[t] + gamma * next_v - vals[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    return adv, adv + vals


def train(env_name='BipedalWalker-v3', hidden_sizes=[256], lr=3e-3, value_lr=3e-3,
          epochs=200, batch_size=5000, gamma=0.99, gae_lambda=0.95, lmbda=0.02,
          update_epochs=4, minibatch_size=512, min_log_std=-0.5, max_log_std=0.0,
          min_log_std_final=None, max_delta=0.5, action_repeat=1,
          resume=None, reward_floor=None, value_warmup=0,
          adv_clip=3.0, clip_eps=0.2, ent_coef=0.01, target_kl=0.03, anneal_lr=True,
          norm='linf', normalize_obs=True, normalize_returns=True,
          render=False, gif_every=20, gif_dir='bipedal_walker_gifs'):
    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name, render_mode='human' if render else None)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Box), \
        "bipedal_walker.py is for continuous (Box) action spaces; use pg_ptorch.py for discrete."

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_low = torch.as_tensor(env.action_space.low, dtype=torch.float32)
    act_high = torch.as_tensor(env.action_space.high, dtype=torch.float32)

    rms = RunningMeanStd(obs_dim) if normalize_obs else None
    # running std of the discounted return, used to rescale rewards (SB3's
    # VecNormalize trick). Keeping the return scale ~O(1) is what lets the value
    # net actually fit the targets; otherwise its huge error (the -100 fall
    # penalty dominates) poisons every GAE advantage and the policy gets no
    # usable signal to escape a local optimum.
    ret_rms = RunningMeanStd(()) if normalize_returns else None

    def env_step(e, action):
        # Action repeat / frame skip: hold one sampled action for `action_repeat`
        # env frames, summing reward, and stop early if the episode ends. One
        # *decision* therefore spans k frames -- the temporal coherence that lets
        # independent per-decision Gaussian noise actually produce sustained limb
        # motion (a gait) instead of jitter that averages to nothing. The policy
        # math is unchanged: each decision is still a clean draw from N(mu, std).
        total = 0.0
        for _ in range(action_repeat):
            obs, rew, term, trunc, info = e.step(action)
            total += rew
            if term or trunc:
                break
        return obs, total, term, trunc, info

    def prep(obs):
        # numpy obs -> normalized float32 tensor
        if rms is not None:
            obs = rms.normalize(obs)
        return torch.as_tensor(obs, dtype=torch.float32)

    # ── policy: diagonal Gaussian ───────────────────────────────────────────────
    # mu(s) flows through the projection layers; log_std is a shared learnable
    # leaf trained by its own projection target.
    policy_net = mlp([obs_dim] + hidden_sizes + [act_dim], norm=norm)
    # log_std is shaped (1, act_dim) rather than (act_dim,) so it is a 2-D
    # parameter ProjectionAdam accepts (Muon only optimizes matrices). The
    # leading batch axis of 1 broadcasts cleanly through the Gaussian op.
    log_std = nn.Parameter(torch.full((1, act_dim), math.log(0.5)))
    policy_params = list(policy_net.parameters()) + [log_std]

    # value baseline (projection-aware), trained toward reward-to-go via MSE target
    value_net = mlp([obs_dim] + hidden_sizes + [1], norm=norm)

    # warm-start: load a saved policy (and its log_std + obs normalizer) and
    # fine-tune from it, instead of relearning to balance from scratch. A policy
    # that already keeps the hull upright is a far better launchpad for
    # discovering a forward gait than random exploration, which on BipedalWalker
    # almost never stumbles into net forward progress before falling.
    if resume:
        ckpt = torch.load(resume, map_location='cpu', weights_only=False)
        policy_net.load_state_dict(ckpt['policy'])
        with torch.no_grad():
            log_std.copy_(ckpt['log_std'])
        if rms is not None and ckpt.get('rms') is not None:
            rms.__dict__.update(ckpt['rms'])
        print('Resumed from %s (saved det_ret %.2f)' % (resume, ckpt.get('det_ret', float('nan'))))

    def get_action(obs_t):
        # sample an action from the current Gaussian policy (no gradient)
        with torch.no_grad():
            mu = policy_net(obs_t)
            a = mu + torch.exp(log_std) * torch.randn_like(mu)
            # log_std's (1, act_dim) shape broadcasts a to (1, act_dim); flatten
            # back to the (act_dim,) vector env.step expects. nan_to_num guards
            # the Box2D sim against a (divergent) NaN action.
            a = torch.clamp(a, act_low, act_high).numpy().reshape(-1)
            return np.nan_to_num(a, nan=0.0)

    policy_opt = ProjectionAdam(policy_params, lr=lr)
    value_opt = ProjectionAdam(value_net.parameters(), lr=value_lr)

    def eval_deterministic(max_steps=1600):
        # Roll out one greedy episode (the Gaussian mean mu(s), no exploration
        # noise) and return its raw return and length. This is the metric that
        # actually matters: the stochastic batch return is dominated by the
        # exploration noise, so it hides whether mu has learned a gait.
        eenv = gym.make(env_name)
        obs, _ = eenv.reset()
        done, steps, ret = False, 0, 0.0
        with torch.no_grad():
            while not done and steps < max_steps:
                mu = policy_net(prep(obs))
                a = torch.clamp(mu, act_low, act_high).numpy().reshape(-1)
                obs, rew, term, trunc, _ = env_step(eenv, np.nan_to_num(a, nan=0.0))
                ret += rew
                done = term or trunc
                steps += 1
        eenv.close()
        return ret, steps

    def record_gif(path, max_steps=1600, frame_stride=2, resize=(200, 300)):
        # Roll out one deterministic episode (the Gaussian mean mu(s), no noise)
        # in an rgb_array env and save it as a GIF. Frames are subsampled
        # (frame_stride) and downscaled (resize) to keep the GIF small.
        renv = gym.make(env_name, render_mode='rgb_array')
        obs, _ = renv.reset()
        frames, done, steps = [], False, 0
        with torch.no_grad():
            while not done and steps < max_steps:
                if steps % frame_stride == 0:
                    frame = np.asarray(renv.render())
                    if resize is not None:
                        frame = np.asarray(Image.fromarray(frame).resize(
                            (resize[1], resize[0]), Image.BILINEAR))
                    frames.append(frame)
                mu = policy_net(prep(obs))
                a = torch.clamp(mu, act_low, act_high).numpy()
                obs, _, term, trunc, _ = env_step(renv, a)
                done = term or trunc
                steps += 1
        renv.close()
        imageio.mimsave(path, frames, fps=30, loop=0)
        return steps

    def train_one_epoch(cur_min_log_std, freeze_policy=False):
        batch_obs, batch_acts, batch_rews = [], [], []
        episodes = []           # (start_idx, length, bootstrap_obs or None)
        batch_rets, batch_lens = [], []   # episode returns / lengths (logging)
        disc_rets = []          # running discounted return per step (for ret_rms)

        obs, _ = env.reset()
        ep_start, ep_rews = 0, []
        disc_ret = 0.0          # gamma-discounted accumulator, reset per episode
        finished_rendering_this_epoch = False

        # ── collect a batch of on-policy transitions ────────────────────────────
        while True:
            if (not finished_rendering_this_epoch) and render:
                env.render()

            batch_obs.append(obs.copy())
            act = get_action(prep(obs))
            next_obs, rew, terminated, truncated, _ = env_step(env, act)
            # clip the -100 fall spike during *training* only (eval/gif keep the
            # true reward): it dominates the value function, so the baseline
            # models "survive vs fall" and the small per-step forward-progress
            # reward gets buried. Softening it lets the agent experiment with
            # gaits (risking a fall) instead of freezing to avoid the cliff.
            if reward_floor is not None and rew < reward_floor:
                rew = reward_floor

            batch_acts.append(act)
            batch_rews.append(rew)
            ep_rews.append(rew)
            disc_ret = gamma * disc_ret + rew
            disc_rets.append(disc_ret)
            obs = next_obs

            if terminated or truncated:
                batch_rets.append(sum(ep_rews))
                batch_lens.append(len(ep_rews))
                # bootstrap value on time-limit truncation, but not on a real
                # terminal (fall): keep the next_obs so we can evaluate V(next).
                episodes.append((ep_start, len(ep_rews),
                                 None if terminated else next_obs.copy()))
                obs, _ = env.reset()
                ep_start, ep_rews = len(batch_obs), []
                disc_ret = 0.0
                finished_rendering_this_epoch = True
                if len(batch_obs) > batch_size:
                    break

        # ── advantages / value targets via GAE (params frozen at collection) ────
        batch_obs = np.asarray(batch_obs, dtype=np.float32)
        if rms is not None:
            rms.update(batch_obs)
        obs_t = prep(batch_obs)
        acts = torch.as_tensor(np.asarray(batch_acts), dtype=torch.float32)
        with torch.no_grad():
            vals = value_net(obs_t).squeeze(-1).numpy()

        # rescale rewards by the running std of the discounted return so value
        # targets land ~O(1). No mean-centering (that would shift the policy
        # objective); just a scale so the value net can fit and GAE advantages
        # stay informative across the -100 fall penalty. The divisor is floored
        # at 1.0 so this can only ever *shrink* the reward scale (taming the
        # -100), never amplify it: when the policy is consistent (e.g. a stable
        # warm-started walker) the return variance collapses, and dividing by a
        # tiny std would blow rewards up and diverge the value net.
        batch_rews = np.asarray(batch_rews, dtype=np.float32)
        if ret_rms is not None:
            ret_rms.update(np.asarray(disc_rets, dtype=np.float32))
            batch_rews = batch_rews / max(np.sqrt(ret_rms.var + 1e-8), 1.0)

        adv = np.zeros(len(batch_obs), dtype=np.float32)
        ret = np.zeros(len(batch_obs), dtype=np.float32)
        for start, length, boot in episodes:
            sl = slice(start, start + length)
            if boot is None:
                last_v = 0.0
            else:
                with torch.no_grad():
                    last_v = float(value_net(prep(boot)).item())
            a, r = compute_gae(np.asarray(batch_rews[start:start + length], dtype=np.float32),
                               vals[sl], last_v, gamma, gae_lambda)
            adv[sl], ret[sl] = a, r

        # standardize advantages, then clip the magnitude: this bounds the
        # proximal step  lmbda * adv * (a-mu)/std^2  in the Gaussian op, the main
        # source of divergence without a PPO ratio-clip to fall back on.
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        adv = np.clip(adv, -adv_clip, adv_clip)
        adv_t = torch.as_tensor(adv)
        ret_t = torch.as_tensor(ret)

        # log-prob under the *collection* policy (pi_old), fixed for this batch —
        # the denominator of the PPO importance ratio.
        with torch.no_grad():
            logp_old = gaussian_logp(acts, policy_net(obs_t), log_std)

        # ── PPO-clipped: several passes of minibatch projection updates ─────────
        N = len(batch_obs)
        idx = np.arange(N)
        pol_acc, val_acc, clipfrac_acc, nb = 0.0, 0.0, 0.0, 0
        stop_update = False
        for _ in range(update_epochs):
            np.random.shuffle(idx)
            kl_acc, kl_nb = 0.0, 0
            for s in range(0, N, minibatch_size):
                mb = torch.as_tensor(idx[s:s + minibatch_size])
                policy_opt.zero_grad()
                value_opt.zero_grad()
                mu = policy_net(obs_t[mb])

                # PPO trust region: gate each sample's projection step by the
                # clip. The probability ratio r = pi_new/pi_old measures how far
                # the policy has drifted from collection. A sample keeps driving
                # the update only while it stays inside the trust region in the
                # direction the advantage wants; once it crosses the band
                # (r>1+eps for A>0, or r<1-eps for A<0) its gradient is clipped
                # away. The effective weight is A*r (PPO's unclipped surrogate
                # coefficient), masked to zero when clipped. r is capped at 1+eps
                # so a drifted negative-advantage sample (whose r is unbounded
                # above) can't drive a runaway proximal step.
                with torch.no_grad():
                    a_mb = adv_t[mb]
                    log_ratio = gaussian_logp(acts[mb], mu, log_std) - logp_old[mb]
                    ratio = torch.exp(log_ratio)
                    unclipped = ((a_mb > 0) & (ratio < 1 + clip_eps)) | \
                                ((a_mb < 0) & (ratio > 1 - clip_eps))
                    w_eff = a_mb * ratio.clamp(max=1.0 + clip_eps) * unclipped.to(ratio.dtype)
                    clipfrac_acc += float((~unclipped).to(ratio.dtype).mean())
                    # unbiased approx KL(pi_old||pi_new) (Schulman): >=0, cheap.
                    kl_acc += float(((ratio - 1.0) - log_ratio).mean())
                    kl_nb += 1

                # policy pseudo-loss -> target for (mu, log_std). One projection
                # step per minibatch; the clip is re-evaluated every pass.
                pol = GaussianPolicyGradientProjection.apply(
                    mu, log_std, acts[mb], w_eff, 1, lmbda, max_delta)
                # value pseudo-loss -> MSE target toward the GAE return
                v = value_net(obs_t[mb]).squeeze(-1)
                val = MSEProjection.apply(v, ret_t[mb])
                (pol + val).backward()
                # entropy bonus: nudge the log_std *target* upward so the
                # diagonal Gaussian keeps exploring instead of collapsing onto
                # the floor and locking into whatever basin it found first. This
                # competes with the op's exploitation pull, settling sigma at an
                # equilibrium -> the main lever on whether a run finds a forward
                # gait vs. a stand-still local optimum. (p.grad holds the target;
                # raising it raises the target the optimizer moves log_std to.)
                if ent_coef and log_std.grad is not None:
                    log_std.grad.add_(ent_coef)
                # value warmup: for the first few epochs after a warm-start the
                # value net is still random, so its advantages are noise that
                # would wreck the loaded policy before the baseline fits. Hold
                # the policy frozen and let only the value net catch up first.
                if not freeze_policy:
                    policy_opt.step()
                value_opt.step()
                # keep sigma in a sane band: a floor so exploration doesn't
                # collapse into the fall-fast local optimum, a ceiling so it
                # can't run away and destabilize the policy.
                with torch.no_grad():
                    log_std.clamp_(min=cur_min_log_std, max=max_log_std)
                pol_acc += float(pol.detach()); val_acc += float(val.detach()); nb += 1
            # KL trust-region early stop: once the policy has drifted too far
            # from the collection policy in a pass, stop updating on this batch.
            # This is the guard against the post-convergence collapse -- without
            # it, the inner loop keeps over-driving a good policy until it falls
            # off the cliff into a degenerate fixed point (identical episodes ->
            # zero advantage variance -> dead gradient).
            if target_kl and kl_nb and (kl_acc / kl_nb) > target_kl:
                stop_update = True
            if stop_update:
                break

        return pol_acc / nb, val_acc / nb, clipfrac_acc / nb, batch_rets, batch_lens

    # gif output dir + a baseline clip of the untrained policy
    os.makedirs(gif_dir, exist_ok=True)
    if gif_every:
        p = os.path.join(gif_dir, 'epoch_0000.gif')
        record_gif(p)
        print('Saved rollout gif -> %s' % p)

    # training loop
    import copy
    hist_loss, hist_ret, hist_len = [], [], []
    best_det = -1e9
    best_state = None
    for i in range(epochs):
        # linear learning-rate anneal (standard PPO): gentle the updates as the
        # policy converges so it refines a found gait instead of over-stepping
        # off the cliff into the post-peak collapse seen with a fixed lr.
        if anneal_lr:
            frac = 1.0 - i / float(epochs)
            for g in policy_opt.param_groups:
                g['lr'] = lr * frac
            for g in value_opt.param_groups:
                g['lr'] = value_lr * frac
        # anneal the log_std floor from min_log_std down to min_log_std_final:
        # start wide for exploration, then let sigma commit to a low-noise gait.
        # Safe to drive low now that max_delta caps the 1/sigma^2 mu step.
        if min_log_std_final is None:
            cur_min_log_std = min_log_std
        else:
            frac = i / float(max(epochs - 1, 1))
            cur_min_log_std = min_log_std + (min_log_std_final - min_log_std) * frac
        # freeze the policy during value warmup so a fresh (random) value net
        # fits the warm-started policy's returns before its advantages are
        # allowed to move the policy.
        freeze_policy = i < value_warmup
        pol_loss, val_loss, clipfrac, batch_rets, batch_lens = train_one_epoch(
            cur_min_log_std, freeze_policy=freeze_policy)
        hist_loss.append(val_loss)
        hist_ret.append(float(np.mean(batch_rets)))
        hist_len.append(float(np.mean(batch_lens)))
        with torch.no_grad():
            ls_mean = log_std.mean().item()
        # deterministic (greedy mu) eval -- the metric that reflects the actual
        # learned gait, free of exploration noise. Run every 10 epochs (one
        # extra rollout) so we can see mu's progress separate from the noisy
        # on-policy batch return. We snapshot the best-by-det_ret policy so a
        # later instability (collapse) can't lose the converged walker.
        det_str = ''
        if i % 10 == 0 or i == epochs - 1:
            det_ret, det_len = eval_deterministic()
            det_str = '\t det_ret: %8.2f \t det_len: %6.1f' % (det_ret, det_len)
            if det_ret > best_det:
                best_det = det_ret
                best_state = (copy.deepcopy(policy_net.state_dict()),
                              log_std.detach().clone(),
                              copy.deepcopy(rms.__dict__) if rms is not None else None)
                det_str += '  <-- best'
        print('epoch: %3d \t pol: %8.2f \t val: %8.2f \t clipfrac: %.2f \t return: %8.2f \t ep_len: %7.1f \t log_std: %.3f%s' %
              (i, pol_loss, val_loss, clipfrac, np.mean(batch_rets), np.mean(batch_lens), ls_mean, det_str))

        # periodically save a deterministic-rollout gif of the current policy
        if gif_every and ((i + 1) % gif_every == 0 or i == epochs - 1):
            p = os.path.join(gif_dir, 'epoch_%04d.gif' % (i + 1))
            record_gif(p)
            print('Saved rollout gif -> %s' % p)

    # restore the best-by-det_ret policy (guards against a late collapse) and
    # save a gif + checkpoint of it -- this is the policy worth keeping.
    if best_state is not None:
        policy_net.load_state_dict(best_state[0])
        with torch.no_grad():
            log_std.copy_(best_state[1])
        if rms is not None and best_state[2] is not None:
            rms.__dict__.update(best_state[2])
        torch.save({'policy': best_state[0], 'log_std': best_state[1],
                    'rms': best_state[2], 'det_ret': best_det},
                   'bipedal_walker_best.pt')
        if gif_every:
            p = os.path.join(gif_dir, 'best_det%.0f.gif' % best_det)
            record_gif(p)
            print('Best deterministic return %.2f -> saved %s + bipedal_walker_best.pt'
                  % (best_det, p))

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
    out_path = 'bipedal_walker_training_%s.png' % env_name.replace('/', '-')
    fig.savefig(out_path, dpi=120)
    print('\nSaved training plot to %s' % out_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='BipedalWalker-v3')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--value_lr', type=float, default=3e-3)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=5000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help='GAE(lambda) advantage smoothing')
    parser.add_argument('--lmbda', type=float, default=0.02,
                        help='proximal/ascent step size for the Gaussian policy target')
    parser.add_argument('--update_epochs', type=int, default=4,
                        help='PPO-style passes over each collected batch')
    parser.add_argument('--minibatch_size', type=int, default=1024)
    parser.add_argument('--min_log_std', type=float, default=-0.5,
                        help='entropy floor (start): lower bound on log_std (exp(-0.5)~=0.61)')
    parser.add_argument('--min_log_std_final', type=float, default=None,
                        help='if set, anneal the log_std floor linearly to this by the '
                             'last epoch -- lets sigma commit to a low-noise gait late in '
                             'training (e.g. -1.5). Default None keeps min_log_std fixed.')
    parser.add_argument('--max_log_std', type=float, default=0.0,
                        help='entropy ceiling: upper bound on log_std (exp(0)=1.0)')
    parser.add_argument('--max_delta', type=float, default=0.5,
                        help='clamp the per-step mu projection move element-wise; bounds the '
                             '1/sigma^2 blow-up at the source (0 or negative disables)')
    parser.add_argument('--action_repeat', type=int, default=1,
                        help='frame skip: hold each sampled action for k env frames. k=4 gives '
                             'the temporal coherence locomotion exploration needs (1 disables)')
    parser.add_argument('--resume', type=str, default=None,
                        help='warm-start: load policy/log_std/rms from this checkpoint and '
                             'fine-tune (e.g. bipedal_walker_best_det9.pt)')
    parser.add_argument('--reward_floor', type=float, default=None,
                        help='clip per-decision training reward to >= this, softening the -100 '
                             'fall penalty so forward-progress reward is not buried (e.g. -10)')
    parser.add_argument('--value_warmup', type=int, default=0,
                        help='freeze the policy for the first N epochs so a fresh value net '
                             'fits before its advantages move a warm-started policy (e.g. 15)')
    parser.add_argument('--adv_clip', type=float, default=3.0,
                        help='clip standardized advantages to +/- this (stability)')
    parser.add_argument('--clip_eps', type=float, default=0.2,
                        help='PPO clip epsilon (trust-region width on the ratio)')
    parser.add_argument('--ent_coef', type=float, default=0.01,
                        help='entropy bonus: upward nudge on the log_std target each step')
    parser.add_argument('--target_kl', type=float, default=0.03,
                        help='KL trust-region: stop inner updates once pi drifts past this (0 disables)')
    parser.add_argument('--no_anneal_lr', dest='anneal_lr', action='store_false',
                        help='disable linear learning-rate annealing')
    parser.add_argument('--hidden', type=int, nargs='+', default=[512])
    parser.add_argument('--norm', type=str, default='linf', choices=['l2', 'linf'])
    parser.add_argument('--gif_every', type=int, default=10,
                        help='save a deterministic-rollout gif every N epochs (0 disables)')
    parser.add_argument('--no_normalize_obs', dest='normalize_obs', action='store_false')
    parser.add_argument('--no_normalize_returns', dest='normalize_returns', action='store_false',
                        help='disable reward rescaling by the running return std')
    parser.set_defaults(normalize_obs=True, normalize_returns=True, anneal_lr=True)
    args = parser.parse_args()
    print('\nProjection policy gradient (ptorch) for continuous control, with value baseline.\n')
    train(env_name=args.env_name, hidden_sizes=args.hidden, lr=args.lr,
          value_lr=args.value_lr, epochs=args.epochs, batch_size=args.batch_size,
          gamma=args.gamma, gae_lambda=args.gae_lambda, lmbda=args.lmbda,
          update_epochs=args.update_epochs, minibatch_size=args.minibatch_size,
          min_log_std=args.min_log_std, max_log_std=args.max_log_std,
          min_log_std_final=args.min_log_std_final,
          max_delta=(args.max_delta if args.max_delta and args.max_delta > 0 else None),
          action_repeat=max(1, args.action_repeat),
          resume=args.resume, reward_floor=args.reward_floor, value_warmup=args.value_warmup,
          adv_clip=args.adv_clip, clip_eps=args.clip_eps, ent_coef=args.ent_coef,
          target_kl=args.target_kl, anneal_lr=args.anneal_lr, norm=args.norm,
          normalize_obs=args.normalize_obs, normalize_returns=args.normalize_returns,
          render=args.render, gif_every=args.gif_every)
