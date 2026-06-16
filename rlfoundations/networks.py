"""Neural-network building blocks for policy-gradient methods.

Shared across Step 2 (REINFORCE) and the actor-critic / PPO steps that follow, so
the same MLP trunk and categorical-policy head get reused instead of rewritten each
time. Pure plumbing — no algorithm logic lives here.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def mlp(sizes: Sequence[int], activation: type[nn.Module] = nn.Tanh) -> nn.Sequential:
    """Build a plain MLP.

    ``sizes = [in_dim, hidden1, ..., out_dim]``. ``activation`` is applied between
    layers but never on the output (we want raw logits / values out).
    """
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class CategoricalPolicy(nn.Module):
    """MLP policy over a discrete action space.

    ``forward`` returns raw logits; ``distribution`` wraps them in a
    ``torch.distributions.Categorical`` you can ``.sample()`` and ``.log_prob()``.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = (128,),
    ) -> None:
        super().__init__()
        self.net = mlp([obs_dim, *hidden_sizes, n_actions])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.forward(obs))
