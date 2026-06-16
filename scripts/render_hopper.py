"""Render a trained Hopper PPO policy to a GIF (showcase artifact).

Loads checkpoints/hopper_ppo.pt (saved by train_hopper.py), replays the greedy
(mean-action) policy in a render_mode='rgb_array' Hopper, and writes media/hopper.gif.
The checkpoint carries the observation normalizer stats — the policy was trained on
normalized observations, so we must normalize here too or it would flail.

    python scripts/render_hopper.py
"""
from __future__ import annotations

import os

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from rlfoundations.networks import GaussianPolicy

CKPT = "checkpoints/hopper_ppo.pt"
OUT = "media/hopper.gif"
MAX_FRAMES = 360          # ~12s at 30 fps; caps GIF size for long episodes
FPS = 30
RESIZE = 320              # downscale 480 -> 320 to keep the GIF README-sized


def main() -> None:
    ckpt = torch.load(CKPT, weights_only=False)
    policy = GaussianPolicy(ckpt["obs_dim"], ckpt["action_dim"], ckpt["hidden_sizes"])
    policy.load_state_dict(ckpt["actor"])
    policy.eval()

    mean, var = ckpt["obs_rms_mean"], ckpt["obs_rms_var"]

    def norm(obs):
        if mean is None:
            return obs
        return (obs - mean) / np.sqrt(var + 1e-8)

    @torch.no_grad()
    def greedy(obs):
        action = policy(torch.as_tensor(norm(obs), dtype=torch.float32)).numpy()
        return np.clip(action, -1.0, 1.0)

    env = gym.make("Hopper-v5", render_mode="rgb_array")
    frames: list = []
    episode = 0
    while len(frames) < MAX_FRAMES:
        obs, _ = env.reset(seed=episode)
        episode += 1
        done = False
        while not done and len(frames) < MAX_FRAMES:
            frames.append(np.array(Image.fromarray(env.render()).resize((RESIZE, RESIZE))))
            obs, _, terminated, truncated, _ = env.step(greedy(obs))
            done = terminated or truncated
    env.close()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    imageio.mimsave(OUT, frames, fps=FPS)
    size_mb = os.path.getsize(OUT) / 1e6
    print(
        f"saved {len(frames)} frames ({len(frames)/FPS:.1f}s @ {FPS}fps, {size_mb:.1f} MB) "
        f"-> {OUT}  (checkpoint final_eval={ckpt.get('final_eval')})"
    )


if __name__ == "__main__":
    main()
