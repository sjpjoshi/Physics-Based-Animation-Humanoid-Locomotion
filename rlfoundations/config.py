"""Run configuration.

A small dataclass-based config so every experiment is reproducible and fully
described by its parameters — the same discipline we'll carry into the training
pipeline in later phases.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class RunConfig:
    # environment
    env_id: str = "CartPole-v1"
    seed: int = 0

    # training budget (used by the algorithm scripts you'll write)
    total_episodes: int = 500

    # Weights & Biases
    wandb_project: str = "rlfoundations"
    wandb_mode: str = "offline"  # "online" after `wandb login`; "disabled" to skip
    run_name: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
