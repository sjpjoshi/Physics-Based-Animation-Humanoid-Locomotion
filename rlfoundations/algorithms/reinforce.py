"""REINFORCE — Monte-Carlo policy gradient (Step 2; YOU implement the core).

The leap from Step 1: Q-learning bootstrapped a value *table* from one-step
transitions. REINFORCE keeps no table and no bootstrap. It rolls out a whole
episode under the current stochastic policy, then shifts the policy parameters to
make each action it took more (or less) likely in proportion to the *return* that
followed it — the score-function / log-derivative trick:

    grad J(theta)  =  E[ sum_t  grad log pi(a_t | s_t) * G_t ]

so we minimize   L = - sum_t  log pi(a_t | s_t) * G_t .

Plumbing here is wired for you: the policy network, action sampling, the rollout
loop, and the optimizer step. The TWO pieces you implement are the heart of the
method and mirror the two halves of that gradient:

    1. compute_returns  ->  the G_t   (discounted returns-to-go)
    2. policy_loss      ->  the  - sum_t log_prob_t * G_t  objective

Both are stubbed with a recipe, exactly like the Step 1 discretizer.
"""
from __future__ import annotations

from typing import List, Sequence

import torch

from rlfoundations.networks import CategoricalPolicy


class REINFORCEAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = (128,),
        lr: float = 1e-2,
        gamma: float = 0.99,
        normalize_returns: bool = True,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            obs_dim:           observation vector size (4 for CartPole).
            n_actions:         number of discrete actions (2 for CartPole).
            hidden_sizes:      MLP hidden-layer widths.
            lr:                Adam learning rate.
            gamma:             discount factor for the returns-to-go.
            normalize_returns: standardize G_t to ~zero-mean/unit-std before the
                               loss (a big variance-reduction win — see update()).
            device:            "cpu" is fine for CartPole.
        """
        self.gamma = gamma
        self.normalize_returns = normalize_returns
        self.device = torch.device(device)

        self.policy = CategoricalPolicy(obs_dim, n_actions, hidden_sizes).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    # --- plumbing: action selection -------------------------------------------
    def _tensor(self, obs) -> torch.Tensor:
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device)

    def select_action(self, obs) -> tuple[int, torch.Tensor]:
        """Sample an action from the current policy.

        Returns (action, log_prob). The log_prob still carries its autograd graph
        so the policy loss can backprop through it — do NOT detach it.
        """
        dist = self.policy.distribution(self._tensor(obs))
        action = dist.sample()
        return int(action.item()), dist.log_prob(action)

    @torch.no_grad()
    def act_greedy(self, obs) -> int:
        """Deterministic argmax action — for evaluation (no exploration)."""
        return int(self.policy(self._tensor(obs)).argmax(dim=-1).item())

    def compute_returns(self, rewards: List[float]) -> torch.Tensor:
        returns: list[float] = []
        running = 0.0

        for reward in reversed(rewards):
            running = float(reward) + self.gamma * running
            returns.append(running)

        returns.reverse()

        return torch.as_tensor(
            returns,
            dtype=torch.float32,
            device=self.device,
        )

    def policy_loss(self, log_probs: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        return -(log_probs * returns).sum()

    # --- plumbing: one update from a full episode -----------------------------
    def update(self, log_probs: List[torch.Tensor], rewards: List[float]) -> float:
        """Run one policy-gradient update from a completed episode.

        Ties your two methods together with the optimizer step. Nothing to edit
        here: it calls compute_returns, optionally standardizes the returns, builds
        the loss via policy_loss, then steps Adam.
        """
        returns = self.compute_returns(rewards)
        logp = torch.stack(log_probs)

        # Variance reduction: standardizing the returns to ~N(0, 1) turns them into
        # a crude advantage (positive = better-than-average action -> push up;
        # negative -> push down) and is often the difference between REINFORCE
        # converging on CartPole and thrashing. Toggle NORMALIZE_RETURNS in the
        # runner to feel the contrast.
        if self.normalize_returns and returns.numel() > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = self.policy_loss(logp, returns)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    # --- plumbing: roll out one episode, then update --------------------------
    def train_episode(self, env, max_steps: int = 10_000) -> float:
        """Roll out one full episode under the current policy, then do one update.

        Returns the episode's total (undiscounted) return.
        """
        obs, _ = env.reset()
        log_probs: List[torch.Tensor] = []
        rewards: List[float] = []
        total = 0.0

        for _ in range(max_steps):
            action, log_prob = self.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)

            log_probs.append(log_prob)
            rewards.append(float(reward))
            total += float(reward)

            if terminated or truncated:
                break

        self.update(log_probs, rewards)
        return total
