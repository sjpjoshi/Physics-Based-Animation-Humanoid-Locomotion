"""Actor-critic — REINFORCE with a learned value baseline (Step 3; YOU implement the core).

The leap from Step 2: REINFORCE scaled each action's log-prob by the raw return
G_t, so the gradient swung with the luck of an episode — high variance, the
climb-then-wobble you watched. Actor-critic adds a second network, the *critic*
V(s), trained to predict the return from each state. Subtracting that learned
baseline leaves the *advantage*

    A_t = G_t - V(s_t)

— "how much better did this action do than the critic expected?" — and we push the
policy by A_t instead of G_t. Actions that merely ride a good state no longer get
credit; only genuine surprises move the policy. Same gradient skeleton, far less
variance. (NORMALIZE_RETURNS in Step 2 was the crude, stateless stand-in for
exactly this baseline.)

Two networks, two losses, two optimizers:

    actor  (CategoricalPolicy, reused from Step 2)  <- L_actor  = -sum_t log pi(a_t|s_t) * A_t
    critic (ValueNetwork, new this step)            <- L_critic = mean_t (V(s_t) - G_t)^2

Plumbing here is wired for you: both networks, action sampling, the rollout (now
also stashing each visited state so the critic can value them), and the twin
optimizer step. The THREE pieces you implement are the conceptual core:

    1. compute_advantages -> A_t = G_t - V(s_t)        (the baseline subtraction)
    2. actor_loss         -> -sum_t log pi * A_t        (policy gradient, baselined)
    3. critic_loss        -> mean (V - G)^2             (regress the critic to the returns)

compute_returns (G_t) is your verified Step-2 code, handed back to you intact.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch

from rlfoundations.networks import CategoricalPolicy, ValueNetwork


class ActorCriticAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = (128,),
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        normalize_advantages: bool = True,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            obs_dim:              observation vector size (4 for CartPole).
            n_actions:            number of discrete actions (2 for CartPole).
            hidden_sizes:         MLP hidden-layer widths (shared shape for both nets).
            actor_lr:             Adam learning rate for the policy.
            critic_lr:            Adam learning rate for the value net.
            gamma:                discount factor for the returns-to-go.
            normalize_advantages: standardize A_t to ~zero-mean/unit-std before the
                                  actor loss (same variance win as Step 2 — see update()).
            device:               "cpu" is fine for CartPole.
        """
        self.gamma = gamma
        self.normalize_advantages = normalize_advantages
        self.device = torch.device(device)

        self.actor = CategoricalPolicy(obs_dim, n_actions, hidden_sizes).to(self.device)
        self.critic = ValueNetwork(obs_dim, hidden_sizes).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    # --- plumbing: action selection -------------------------------------------
    def _tensor(self, x) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def select_action(self, obs) -> tuple[int, torch.Tensor]:
        """Sample an action from the current policy (the actor).

        Returns (action, log_prob). The log_prob still carries its autograd graph
        so the actor loss can backprop through it — do NOT detach it.
        """
        dist = self.actor.distribution(self._tensor(obs))
        action = dist.sample()
        return int(action.item()), dist.log_prob(action)

    @torch.no_grad()
    def act_greedy(self, obs) -> int:
        """Deterministic argmax action — for evaluation (no exploration)."""
        return int(self.actor(self._tensor(obs)).argmax(dim=-1).item())

    def compute_returns(self, rewards: List[float]) -> torch.Tensor:
        """Discounted returns-to-go G_t (your verified Step-2 code, reused intact).

        Still the backbone here: the critic regresses toward these, and the
        advantage is measured against them.
        """
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

    def compute_advantages(
        self, returns: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        """Compute A_t = G_t - V(s_t), detaching V for the actor update."""
        return returns - values.detach()
    
    def actor_loss(
        self, log_probs: torch.Tensor, advantages: torch.Tensor
    ) -> torch.Tensor:
        """Actor loss: push up actions with positive advantage, down with negative."""
        return -(log_probs * advantages).sum()

    def critic_loss(
        self, values: torch.Tensor, returns: torch.Tensor
    ) -> torch.Tensor:
        """Critic loss: regress V(s_t) toward Monte Carlo return G_t."""
        return ((values - returns) ** 2).mean()

    # --- plumbing: one update from a full episode -----------------------------
    def update(
        self,
        states: List,
        log_probs: List[torch.Tensor],
        rewards: List[float],
    ) -> tuple[float, float]:
        """Run one actor-critic update from a completed episode.

        Ties your three methods together (nothing to edit here): compute the
        returns-to-go, value every visited state in one batched critic forward,
        form the advantages, optionally standardize them (same variance trick as
        Step 2), then build and step the two losses on their own optimizers.
        """
        returns = self.compute_returns(rewards)
        states_t = self._tensor(np.asarray(states, dtype=np.float32))
        values = self.critic(states_t)            # V(s_t) for every step, with grad
        logp = torch.stack(log_probs)

        advantages = self.compute_advantages(returns, values)
        if self.normalize_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_l = self.actor_loss(logp, advantages)
        critic_l = self.critic_loss(values, returns)

        # actor and critic are separate networks with disjoint graphs (the advantage
        # is detached), so we can backprop both losses and step both optimizers with
        # no retain_graph juggling.
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_l.backward()
        critic_l.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return float(actor_l.item()), float(critic_l.item())

    # --- plumbing: roll out one episode, then update --------------------------
    def train_episode(self, env, max_steps: int = 10_000) -> float:
        """Roll out one full episode under the current policy, then do one update.

        Now also stashes each visited state (the critic needs to value them in the
        update). Returns the episode's total (undiscounted) return.
        """
        obs, _ = env.reset()
        states: List = []
        log_probs: List[torch.Tensor] = []
        rewards: List[float] = []
        total = 0.0

        for _ in range(max_steps):
            action, log_prob = self.select_action(obs)
            states.append(obs)                    # s_t that produced this action
            log_probs.append(log_prob)
            obs, reward, terminated, truncated, _ = env.step(action)

            rewards.append(float(reward))
            total += float(reward)

            if terminated or truncated:
                break

        self.update(states, log_probs, rewards)
        return total
