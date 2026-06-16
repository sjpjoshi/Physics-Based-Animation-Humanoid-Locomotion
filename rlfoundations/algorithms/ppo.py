from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from rlfoundations.networks import CategoricalPolicy, ValueNetwork


class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_sizes: Sequence[int] = (64, 64),
        lr: float = 2.5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        num_minibatches: int = 4,
        normalize_advantages: bool = True,
        clip_vloss: bool = True,
        orthogonal_init: bool = True,
        device: str = "cpu",
    ) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.num_minibatches = num_minibatches
        self.normalize_advantages = normalize_advantages
        self.clip_vloss = clip_vloss
        self.device = torch.device(device)

        self.actor = CategoricalPolicy(
            obs_dim, n_actions, hidden_sizes, orthogonal_init=orthogonal_init
        ).to(self.device)
        self.critic = ValueNetwork(
            obs_dim, hidden_sizes, orthogonal_init=orthogonal_init
        ).to(self.device)
        self.params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=lr, eps=1e-5)

        # PPO collects rollouts continuously, so the "current" env state persists
        # across collect_rollout() calls (set on the first call).
        self._next_obs = None
        self._next_done = 0.0

    # --- plumbing: tensors + action selection ---------------------------------
    def _tensor(self, x) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def select_action_batch(self, obs):
        """Vectorized rollout-time step: obs [N, obs_dim] -> (actions, log_probs, values),
        each a length-N numpy array.

        No grad — these are the *behaviour-policy* outputs stored in the buffer (the
        "old" log-probs PPO clips against, and V(s_t) for GAE). One batched forward
        through actor + critic serves all N envs at once (the throughput win).
        """
        obs_t = self._tensor(obs)
        dist = self.actor.distribution(obs_t)
        actions = dist.sample()
        return (
            actions.cpu().numpy(),
            dist.log_prob(actions).cpu().numpy(),
            self.critic(obs_t).cpu().numpy(),
        )

    @torch.no_grad()
    def act_greedy(self, obs) -> int:
        """Deterministic argmax action — for evaluation."""
        return int(self.actor(self._tensor(obs)).argmax(dim=-1).item())

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-score stored (obs, action) pairs under the CURRENT policy, WITH grad.

        Returns (new_log_probs, entropy, values), each shape [B]. This is what makes
        PPO's ratio differentiable through the new policy while the old log-probs
        stay constant.
        """
        dist = self.actor.distribution(obs)
        return dist.log_prob(actions), dist.entropy(), self.critic(obs)
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: float,
        next_done: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards)

        # Convert next_value / next_done to tensors on the right device.
        next_value = torch.as_tensor(
        next_value,
        dtype=values.dtype,
        device=values.device,
        )

        next_done = torch.as_tensor(
        next_done,
        dtype=dones.dtype,
        device=dones.device,
        )

        last_gae = torch.zeros_like(rewards[0])
        for t in reversed(range(rewards.shape[0])):
            if t == rewards.shape[0] - 1:
                next_nonterminal = 1.0 - next_done
                next_values = next_value
            else:
                next_nonterminal = 1.0 - dones[t + 1]
                next_values = values[t + 1]

            delta = (
                rewards[t]
                + self.gamma * next_values * next_nonterminal
                - values[t]
            )

            last_gae = (
                delta
                + self.gamma
                * self.gae_lambda
                * next_nonterminal
                * last_gae
            )

            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def ppo_policy_loss(
        self,
        new_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """PPO clipped policy loss."""
        ratio = torch.exp(new_log_probs - old_log_probs)

        unclipped = ratio * advantages
        clipped = torch.clamp(
            ratio,
            1.0 - self.clip_coef,
            1.0 + self.clip_coef,
        ) * advantages

        return -torch.min(unclipped, clipped).mean()

    # --- plumbing: collect a fixed-horizon rollout over N parallel envs --------
    def collect_rollout(self, envs, num_steps: int) -> tuple[dict, list[float]]:
        """Step each of the N envs `num_steps` times under the current policy.

        Hand-rolled vector (envs is a plain list): we step every env and reset it
        the instant it's done, so we keep full control of the truncation bootstrap
        (gamma * V(s_T) folded into the reward, exactly as single-env) and never
        touch Gymnasium's NEXT_STEP autoreset. Buffers are [num_steps, N]; episode
        boundaries live per-env in `dones` (CleanRL convention: dones[t] = 1 if s_t
        was a fresh-reset state). Continuous across iterations — env state persists
        on the agent. Returns (buffer, completed-episode returns for logging).
        """
        n_envs = len(envs)
        obs_dim = envs[0].observation_space.shape[0]

        if self._next_obs is None:
            self._next_obs = np.stack([env.reset()[0] for env in envs]).astype(np.float32)
            self._next_done = np.zeros(n_envs, dtype=np.float32)

        obs_buf = np.zeros((num_steps, n_envs, obs_dim), dtype=np.float32)
        actions_buf = np.zeros((num_steps, n_envs), dtype=np.int64)
        logp_buf = np.zeros((num_steps, n_envs), dtype=np.float32)
        rewards_buf = np.zeros((num_steps, n_envs), dtype=np.float32)
        dones_buf = np.zeros((num_steps, n_envs), dtype=np.float32)
        values_buf = np.zeros((num_steps, n_envs), dtype=np.float32)

        ep_returns: list[float] = []
        ep_return = np.zeros(n_envs, dtype=np.float32)

        for t in range(num_steps):
            obs_buf[t] = self._next_obs
            dones_buf[t] = self._next_done

            actions, logps, values = self.select_action_batch(self._next_obs)
            actions_buf[t] = actions
            logp_buf[t] = logps
            values_buf[t] = values

            for i, env in enumerate(envs):
                next_obs_i, reward, terminated, truncated, _ = env.step(int(actions[i]))
                rewards_buf[t, i] = reward
                ep_return[i] += reward

                # Per-env truncation bootstrap (same fix as single-env): a time-limit
                # truncation is NOT a real terminal, so fold gamma * V(s_T) into the
                # reward before resetting; GAE still masks the boundary via dones.
                if truncated and not terminated:
                    with torch.no_grad():
                        rewards_buf[t, i] += self.gamma * float(self.critic(self._tensor(next_obs_i)).item())

                done = terminated or truncated
                if done:
                    ep_returns.append(float(ep_return[i]))
                    ep_return[i] = 0.0
                    next_obs_i, _ = env.reset()

                self._next_obs[i] = next_obs_i
                self._next_done[i] = float(done)

        with torch.no_grad():
            next_value = self.critic(self._tensor(self._next_obs)).cpu().numpy()  # [N]

        buffer = {
            "obs": obs_buf,
            "actions": actions_buf,
            "log_probs": logp_buf,
            "rewards": rewards_buf,
            "dones": dones_buf,
            "values": values_buf,
            "next_value": next_value,
            "next_done": self._next_done.copy(),
        }
        return buffer, ep_returns

    # --- plumbing: PPO update over the collected rollout ----------------------
    def update(self, buffer: dict) -> dict:
        """K epochs of minibatch SGD on the clipped objective (nothing to edit).

        Computes GAE once on the whole rollout (your compute_gae), then repeatedly
        shuffles + slices the batch, re-scores each minibatch under the current
        policy, and steps the combined loss. Returns last-minibatch stats for logs.
        """
        obs = self._tensor(buffer["obs"])                  # [T, N, obs_dim]
        actions = torch.as_tensor(buffer["actions"], dtype=torch.long, device=self.device)  # [T, N]
        old_log_probs = self._tensor(buffer["log_probs"])  # [T, N]
        rewards = self._tensor(buffer["rewards"])           # [T, N]
        dones = self._tensor(buffer["dones"])               # [T, N]
        values = self._tensor(buffer["values"])             # [T, N]

        # GAE runs on the [T, N] rollout (your compute_gae broadcasts over the env axis)
        advantages, returns = self.compute_gae(
            rewards, values, dones, buffer["next_value"], buffer["next_done"]
        )

        # flatten [T, N] -> [T*N]: the env axis folds into the batch for minibatch SGD
        obs = obs.reshape(-1, obs.shape[-1])
        actions = actions.reshape(-1)
        old_log_probs = old_log_probs.reshape(-1)
        values = values.reshape(-1)
        advantages = advantages.reshape(-1)
        returns = returns.reshape(-1)

        batch_size = obs.shape[0]
        minibatch_size = batch_size // self.num_minibatches
        idx = np.arange(batch_size)

        stats = {"actor_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        for _ in range(self.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, batch_size, minibatch_size):
                mb = idx[start : start + minibatch_size]

                new_log_probs, entropy, new_values = self.evaluate_actions(obs[mb], actions[mb])

                mb_adv = advantages[mb]
                if self.normalize_advantages and mb_adv.numel() > 1:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                actor_l = self.ppo_policy_loss(new_log_probs, old_log_probs[mb], mb_adv)

                # value loss, optionally clipped to a trust region around the old
                # value predictions (the critic's analogue of the policy clip)
                if self.clip_vloss:
                    mb_values = values[mb]
                    v_unclipped = (new_values - returns[mb]) ** 2
                    v_clipped_pred = mb_values + torch.clamp(
                        new_values - mb_values, -self.clip_coef, self.clip_coef
                    )
                    v_clipped = (v_clipped_pred - returns[mb]) ** 2
                    value_l = 0.5 * torch.max(v_unclipped, v_clipped).mean()
                else:
                    value_l = 0.5 * ((new_values - returns[mb]) ** 2).mean()

                entropy_l = -entropy.mean()  # maximize entropy -> minimize its negative

                loss = actor_l + self.vf_coef * value_l + self.ent_coef * entropy_l

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
                self.optimizer.step()

                stats = {
                    "actor_loss": float(actor_l.item()),
                    "value_loss": float(value_l.item()),
                    "entropy": float(entropy.mean().item()),
                }
        return stats
