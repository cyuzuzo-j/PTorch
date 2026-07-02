import torch
import torch.nn as nn
import numpy as np
try:
    import gymnasium as gym          # maintained drop-in, supports NumPy 2.0
except ImportError:
    import gym                       # fall back to legacy gym
# pull spaces from whichever module 'gym' actually is (gymnasium or legacy gym)
Discrete, Box = gym.spaces.Discrete, gym.spaces.Box
import matplotlib
matplotlib.use('Agg')  # headless-safe backend (no display on the cluster)
import matplotlib.pyplot as plt

# ── ptorch (cyclic projections) ──────────────────────────────────────────────
# Drop-in replacement for the torch building blocks used by pg.py:
#   nn.Linear  -> ptorch.nn.modules.Linear   (projection-aware)
#   nn.Tanh    -> ptorch.nn.modules.ReLU     (ptorch has no Tanh; ReLU is the
#                                             available projection activation)
#   Adam       -> ptorch.optim_static.ProjectionAdam
import ptorch  # noqa: F401  (applies the projection overrides on import)
from ptorch.nn.modules import Linear, ReLU
from ptorch import config as ptorch_config
from ptorch.optim_static import ProjectionAdam
from ptorch.core.ops import PolicyGradientProjection


def mlp(sizes, norm='linf'):
    # Build a feedforward projection network (Linear + ReLU, no head activation).
    layers = []
    for j in range(len(sizes)-1):
        layers.append(Linear(sizes[j], sizes[j+1], norm=norm))
        if j < len(sizes)-2:
            layers.append(ReLU(norm=norm))
    return nn.Sequential(*layers)

def train(env_name='LunarLander-v3', hidden_sizes=[64, 64], lr=1e-2,
          epochs=200, batch_size=5000, norm='linf', render=False):
    # LunarLander-v3 is a markedly harder benchmark than CartPole: an 8-dim
    # continuous observation, 4 discrete actions, and a shaped reward that runs
    # negative on crashes (solving threshold ~200). It therefore needs more
    # capacity (two hidden layers) and many more epochs of vanilla REINFORCE.
    # Requires the Box2D physics backend: `pip install "gymnasium[box2d]"`.

    # make environment, check spaces, get obs / act dims
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # make core of policy network
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts], norm=norm)

    # make action selection function (outputs int actions, sampled from policy).
    # Sampling carries no gradient, so no Categorical / projection is needed —
    # just draw from softmax(logits) under no_grad.
    def get_action(obs):
        with torch.no_grad():
            probs = torch.softmax(logits_net(obs), dim=-1)
            return torch.multinomial(probs, 1).item()

    # projection-based policy-gradient pseudo-loss. The backward returns a
    # *target* for the logits (proximal REINFORCE step), not a gradient — the
    # projection-aware Linear/ReLU layers and ProjectionAdam consume it.
    def compute_loss(obs, act, weights):
        logits = logits_net(obs)
        return PolicyGradientProjection.apply(logits, act, weights)

    # make optimizer (projection-aware Adam; pseudo-grads are ~parameter-scaled,
    # so a larger lr than Adam's 1e-3 default is appropriate)
    optimizer = ProjectionAdam(logits_net.parameters(), lr=lr)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs, _ = env.reset()    # gym>=0.26 returns (obs, info)
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, terminated, truncated, _ = env.step(act)  # gym>=0.26 step API
            done = terminated or truncated

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                obs, _ = env.reset()
                done, ep_rews = False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()
        # Standardize returns so the proximal step size (lambda * weight) stays
        # O(lambda) regardless of the raw return scale (LunarLander returns span
        # roughly -300..+300, so this centering/whitening matters even more here).
        weights = torch.as_tensor(batch_weights, dtype=torch.float32)
        weights = (weights - weights.mean()) / (weights.std() + 1e-8)
        batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                                  act=torch.as_tensor(batch_acts, dtype=torch.int32),
                                  weights=weights
                                  )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens

    # training loop
    hist_loss, hist_ret, hist_len = [], [], []
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        hist_loss.append(batch_loss.item())
        hist_ret.append(float(np.mean(batch_rets)))
        hist_len.append(float(np.mean(batch_lens)))
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

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
    axes[1].set_title('Policy-gradient pseudo-loss')
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = 'pg_ptorch_training_%s.png' % env_name.replace('/', '-')
    fig.savefig(out_path, dpi=120)
    print('\nSaved training plot to %s' % out_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='LunarLander-v3')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--norm', type=str, default='l2', choices=['l2', 'linf'])
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient (ptorch projections).\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr, norm=args.norm)
