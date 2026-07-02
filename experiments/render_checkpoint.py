"""Render a saved bipedal_walker checkpoint to a GIF (deterministic mu rollout).

Usage:  python render_checkpoint.py <checkpoint.pt> [out.gif]
"""
import sys
import numpy as np
import torch
import gymnasium as gym
import imageio.v2 as imageio
from PIL import Image

from bipedal_walker import mlp, RunningMeanStd  # reuse the exact architecture

ckpt_path = sys.argv[1] if len(sys.argv) > 1 else 'bipedal_walker_best.pt'
out_path = sys.argv[2] if len(sys.argv) > 2 else ckpt_path.replace('.pt', '.gif')

ckpt = torch.load(ckpt_path, weights_only=False)
env = gym.make('BipedalWalker-v3', render_mode='rgb_array')
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
act_low = torch.as_tensor(env.action_space.low, dtype=torch.float32)
act_high = torch.as_tensor(env.action_space.high, dtype=torch.float32)

# rebuild policy (norm='l2' matches the sweep) and obs normalizer
policy_net = mlp([obs_dim, 256, 256, act_dim], norm='l2')
policy_net.load_state_dict(ckpt['policy'])
rms = RunningMeanStd(obs_dim)
if ckpt.get('rms') is not None:
    rms.__dict__.update(ckpt['rms'])


def prep(o):
    return torch.as_tensor(rms.normalize(o), dtype=torch.float32)


obs, _ = env.reset()
frames, done, steps, ret = [], False, 0, 0.0
with torch.no_grad():
    while not done and steps < 1600:
        if steps % 2 == 0:
            f = np.asarray(env.render())
            f = np.asarray(Image.fromarray(f).resize((300, 200), Image.BILINEAR))
            frames.append(f)
        mu = policy_net(prep(obs))
        a = torch.clamp(mu, act_low, act_high).numpy().reshape(-1)
        obs, rew, term, trunc, _ = env.step(np.nan_to_num(a, nan=0.0))
        ret += rew
        done = term or trunc
        steps += 1
env.close()
imageio.mimsave(out_path, frames, fps=30, loop=0)
print('return=%.2f  steps=%d  ->  %s' % (ret, steps, out_path))
