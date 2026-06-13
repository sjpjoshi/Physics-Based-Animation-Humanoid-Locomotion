"""Step 0 smoke test: confirm the env + torch + W&B stack works end to end.

Runs a RANDOM policy on CartPole for a few episodes and logs the returns to
Weights & Biases (offline by default). No learning here — this just proves the
plumbing before we start writing algorithms.

Run from the repo root:
    python scripts/smoke_test.py
"""
from __future__ import annotations

import numpy as np

from rlfoundations.config import RunConfig
from rlfoundations.envs import make_env
from rlfoundations.utils.logging import Logger
from rlfoundations.utils.seeding import set_seed


def main() -> None:
    cfg = RunConfig(total_episodes=10, run_name="smoke-test")

    # Report the stack so we can see versions/devices at a glance.
    import gymnasium
    import torch

    print(f"torch     : {torch.__version__} (cuda available: {torch.cuda.is_available()})")
    print(f"gymnasium : {gymnasium.__version__}")
    print(f"env       : {cfg.env_id}")

    set_seed(cfg.seed)
    env = make_env(cfg.env_id, seed=cfg.seed)

    returns: list[float] = []
    with Logger(cfg.wandb_project, cfg.as_dict(), mode=cfg.wandb_mode, run_name=cfg.run_name) as logger:
        for episode in range(cfg.total_episodes):
            obs, _ = env.reset()
            done = False
            ep_return = 0.0
            while not done:
                action = env.action_space.sample()  # random policy
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_return += float(reward)
                done = terminated or truncated
            returns.append(ep_return)
            logger.log({"episode_return": ep_return}, step=episode)
            print(f"episode {episode:2d}  return {ep_return:6.1f}")

    env.close()
    print(f"\nmean return over {cfg.total_episodes} random episodes: {np.mean(returns):.1f}")
    print("Smoke test OK: env, torch, and W&B logging all work.")


if __name__ == "__main__":
    main()
