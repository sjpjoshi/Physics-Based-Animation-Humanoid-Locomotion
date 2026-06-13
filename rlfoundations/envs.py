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
