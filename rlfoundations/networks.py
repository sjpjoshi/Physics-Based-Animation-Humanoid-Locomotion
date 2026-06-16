"""Neural-network building blocks for policy-gradient methods.

Shared across Step 2 (REINFORCE) and the actor-critic / PPO steps that follow, so
the same MLP trunk and categorical-policy head get reused instead of rewritten each
time. Pure plumbing — no algorithm logic lives here.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def mlp(
    sizes: Sequence[int],
    activation: type[nn.Module] = nn.Tanh,
    orthogonal: bool = False,
    output_gain: float = 2.0 ** 0.5,
) -> nn.Sequential:
    """Build a plain MLP.

    ``sizes = [in_dim, hidden1, ..., out_dim]``. ``activation`` is applied between
    layers but never on the output (we want raw logits / values out).

    When ``orthogonal`` is set, weights get the PPO-standard orthogonal init
    (CleanRL): hidden layers with gain sqrt(2), the output layer with ``output_gain``
    (tiny for a policy head so the initial policy stays ~uniform; 1.0 for a value
    head), and all biases zeroed. Default off, so REINFORCE/actor-critic are unchanged.
    """
    layers: list[nn.Module] = []
    n_layers = len(sizes) - 1
    for i in range(n_layers):
        linear = nn.Linear(sizes[i], sizes[i + 1])
        if orthogonal:
            gain = output_gain if i == n_layers - 1 else 2.0 ** 0.5
            nn.init.orthogonal_(linear.weight, gain)
            nn.init.zeros_(linear.bias)
        layers.append(linear)
        if i < n_layers - 1:
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
        orthogonal_init: bool = False,
    ) -> None:
        super().__init__()
        self.net = mlp(
            [obs_dim, *hidden_sizes, n_actions],
            orthogonal=orthogonal_init,
            output_gain=0.01,  # small policy-head gain -> near-uniform initial policy
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.forward(obs))


class ValueNetwork(nn.Module):
    """MLP state-value critic V(s).

    A scalar value head on the same MLP trunk the policy uses. ``forward`` squeezes
    off the trailing size-1 dimension, so a batch of observations ``[T, obs_dim]``
    maps to a flat ``[T]`` of values (and a single observation to a 0-d scalar) —
    shaped to line up directly with the returns-to-go in the actor-critic update.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: Sequence[int] = (128,),
        orthogonal_init: bool = False,
    ) -> None:
        super().__init__()
        self.net = mlp(
            [obs_dim, *hidden_sizes, 1],
            orthogonal=orthogonal_init,
            output_gain=1.0,  # value-head gain
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)
