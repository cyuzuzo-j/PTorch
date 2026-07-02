"""Train the projection policy-gradient agent on LunarLander and save a GIF.

Trains the same ptorch (cyclic-projection) policy used by pg_ptorch.py for a
budget of epochs, then rolls out a few greedy episodes under render_mode=
'rgb_array' and writes the frames to an animated GIF with PIL.

  python pg_gif.py --epochs 80 --episodes 3 --out pg_lunarlander.gif

Requires the Box2D backend: pip install "gymnasium[box2d]".
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

try:
    import gymnasium as gym
except ImportError:
    import gym

# ptorch projection building blocks (overrides applied on import).
import ptorch  # noqa: F401
from ptorch.nn.modules import Linear, ReLU
from ptorch.core.ops import PolicyGradientProjection
from ptorch.optim_static import ProjectionAdam

HERE = os.path.dirname(os.path.abspath(__file__))


def mlp(sizes, norm="l2"):
    layers = []
    for j in range(len(sizes) - 1):
        layers.append(Linear(sizes[j], sizes[j + 1], norm=norm))
        if j < len(sizes) - 2:
            layers.append(ReLU(norm=norm))
    return nn.Sequential(*layers)


def train(env_name, hidden, lr, epochs, batch_size, norm, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n
    net = mlp([obs_dim] + hidden + [n_acts], norm=norm)
    optimizer = ProjectionAdam(net.parameters(), lr=lr)

    def get_action(obs):
        with torch.no_grad():
            probs = torch.softmax(net(obs), dim=-1)
            return torch.multinomial(probs, 1).item()

    # Vanilla REINFORCE on LunarLander is unstable at useful learning rates: a
    # run often climbs, then collapses and never recovers (the very effect the
    # stability benchmark measures). So snapshot the best-performing policy and
    # record the GIF from that, instead of from whatever the final epoch left.
    import copy
    best_ret = -float("inf")
    best_state = copy.deepcopy(net.state_dict())

    for i in range(epochs):
        batch_obs, batch_acts, batch_weights, batch_rets = [], [], [], []
        obs, _ = env.reset()
        ep_rews = []
        while True:
            batch_obs.append(obs.copy())
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, terminated, truncated, _ = env.step(act)
            batch_acts.append(act)
            ep_rews.append(rew)
            if terminated or truncated:
                batch_rets.append(sum(ep_rews))
                batch_weights += [sum(ep_rews)] * len(ep_rews)
                obs, _ = env.reset()
                ep_rews = []
                if len(batch_obs) > batch_size:
                    break

        weights = torch.as_tensor(np.array(batch_weights), dtype=torch.float32)
        weights = (weights - weights.mean()) / (weights.std() + 1e-8)
        optimizer.zero_grad()
        loss = PolicyGradientProjection.apply(
            net(torch.as_tensor(np.array(batch_obs), dtype=torch.float32)),
            torch.as_tensor(np.array(batch_acts), dtype=torch.int64),
            weights,
        )
        loss.backward()
        optimizer.step()
        mean_ret = float(np.mean(batch_rets))
        if mean_ret > best_ret:
            best_ret = mean_ret
            best_state = copy.deepcopy(net.state_dict())
        print("epoch %3d  return %8.1f%s"
              % (i, mean_ret, "  *best*" if mean_ret == best_ret else ""), flush=True)

    env.close()
    net.load_state_dict(best_state)
    print("\nrecording from best policy (train mean return %.1f)" % best_ret, flush=True)
    return net


def record_gif(net, env_name, out_path, episodes, max_frames, stride, seed):
    env = gym.make(env_name, render_mode="rgb_array")
    frames = []
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + 1000 + ep)
        done, ep_ret, t = False, 0.0, 0
        while not done:
            if t % stride == 0:
                frames.append(Image.fromarray(env.render()))
            with torch.no_grad():
                act = int(torch.softmax(
                    net(torch.as_tensor(obs, dtype=torch.float32)), dim=-1).argmax())
            obs, rew, terminated, truncated, _ = env.step(act)
            ep_ret += rew
            done = terminated or truncated
            t += 1
            if len(frames) >= max_frames:
                break
        returns.append(ep_ret)
        print("recorded episode %d  return %.1f  frames %d"
              % (ep, ep_ret, len(frames)), flush=True)
        if len(frames) >= max_frames:
            break
    env.close()

    frames[0].save(
        out_path, save_all=True, append_images=frames[1:],
        duration=40, loop=0, optimize=True,
    )
    print("\nSaved GIF (%d frames) -> %s  | greedy returns: %s"
          % (len(frames), out_path, ["%.0f" % r for r in returns]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="LunarLander-v3")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=4000)
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--norm", default="l2", choices=["l2", "linf"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max_frames", type=int, default=600)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--out", default="pg_lunarlander.gif")
    args = p.parse_args()

    net = train(args.env, args.hidden, args.lr, args.epochs,
                args.batch_size, args.norm, args.seed)
    out_path = args.out if os.path.isabs(args.out) else os.path.join(HERE, args.out)
    record_gif(net, args.env, out_path, args.episodes,
               args.max_frames, args.stride, args.seed)


if __name__ == "__main__":
    main()
