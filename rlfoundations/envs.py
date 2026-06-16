"""Gymnasium environment factory.

Centralizes environment creation + seeding so every script builds envs the same
way. Vectorized environments will be added here in a later step — they become
important once we move past CartPole to MuJoCo tasks.
"""
from __future__ import annotations

import gymnasium as gym


def make_env(env_id: str, seed: int | None = None, render_mode: str | None = None) -> gym.Env:
    """Create a single Gymnasium environment, seeded for reproducibility."""
    env = gym.make(env_id, render_mode=render_mode)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def make_vector_env(
    env_id: str,
    num_envs: int,
    seed: int | None = None,
    render_mode: str | None = None,
) -> list[gym.Env]:
    """Create a list of `num_envs` independent envs, each seeded distinctly.

    Hand-rolled vector (a plain list, NOT gym.vector.SyncVectorEnv): the PPO rollout
    steps and resets each env itself, which keeps the truncation bootstrap clean and
    sidesteps Gymnasium 1.0+ NEXT_STEP autoreset. Distinct per-env seeds (seed + i)
    keep the run reproducible while decorrelating the parallel rollouts.
    """
    return [
        make_env(env_id, seed=None if seed is None else seed + i, render_mode=render_mode)
        for i in range(num_envs)
    ]
