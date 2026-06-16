from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from rlfoundations.networks import CategoricalPolicy, GaussianPolicy, ValueNetwork


class RunningMeanStd:
    """Running mean/variance (Welford parallel / Chan update) for obs normalization.

    MuJoCo observations span very different scales (positions ~1, velocities much
    larger), which makes a single MLP struggle. PPO standard practice (CleanRL, SB3)
    normalizes observations by a running estimate of their mean/std. We keep it on
    the agent so the SAME statistics normalize both rollout and eval — no separate
    per-env wrapper to keep in sync.
    """

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-4) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean) / np.sqrt(self.var + 1e-8)


class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int | None = None,
        action_dim: int | None = None,
        continuous: bool = False,
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
        normalize_obs: bool = False,
        normalize_reward: bool = False,
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
        self.continuous = continuous
        self.action_dim = action_dim
        self.normalize_obs = normalize_obs
        self.obs_rms = RunningMeanStd((obs_dim,)) if normalize_obs else None
        self.normalize_reward = normalize_reward
        self.return_rms = RunningMeanStd(()) if normalize_reward else None
        self.device = torch.device(device)

        if continuous:
            self.actor = GaussianPolicy(
                obs_dim, action_dim, hidden_sizes, orthogonal_init=orthogonal_init
            ).to(self.device)
        else:
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

    def _norm_obs(self, obs):
        """Normalize observations with the running mean/std (no-op if disabled).

        Applied everywhere the nets see an observation — rollout, truncation
        bootstrap, eval — so all three share one normalizer. Does NOT update the
        stats; collect_rollout owns the (training-only) updates, so eval uses frozen
        stats.
        """
        if self.obs_rms is None:
            return obs
        return self.obs_rms.normalize(obs).astype(np.float32)

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
    def act_greedy(self, obs):
        """Deterministic action for evaluation: argmax (discrete) or the mean (continuous).

        For continuous, GaussianPolicy's forward IS the mean; we clip it to the env's
        [-1, 1] action box before returning (the same clip the rollout applies).
        """
        obs = self._norm_obs(obs)  # frozen stats — eval shares the training normalizer
        if self.continuous:
            mean = self.actor(self._tensor(obs))
            return np.clip(mean.cpu().numpy(), -1.0, 1.0)
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
            self._ret_acc = np.zeros(n_envs, dtype=np.float64)  # discounted-return accumulator (reward norm)

        obs_buf = np.zeros((num_steps, n_envs, obs_dim), dtype=np.float32)
        if self.continuous:
            actions_buf = np.zeros((num_steps, n_envs, self.action_dim), dtype=np.float32)
        else:
            actions_buf = np.zeros((num_steps, n_envs), dtype=np.int64)
        logp_buf = np.zeros((num_steps, n_envs), dtype=np.float32)
        rewards_buf = np.zeros((num_steps, n_envs), dtype=np.float32)
        dones_buf = np.zeros((num_steps, n_envs), dtype=np.float32)
        values_buf = np.zeros((num_steps, n_envs), dtype=np.float32)

        ep_returns: list[float] = []
        ep_return = np.zeros(n_envs, dtype=np.float32)

        for t in range(num_steps):
            # update obs stats with the RAW obs, then normalize for storage + policy.
            # Storing the normalized obs keeps update() consistent (it re-feeds them
            # without re-normalizing) — same contract as CleanRL's obs wrapper.
            if self.obs_rms is not None:
                self.obs_rms.update(self._next_obs)
            norm_obs = self._norm_obs(self._next_obs)

            obs_buf[t] = norm_obs
            dones_buf[t] = self._next_done

            actions, logps, values = self.select_action_batch(norm_obs)
            actions_buf[t] = actions
            logp_buf[t] = logps
            values_buf[t] = values

            for i, env in enumerate(envs):
                # discrete: int action; continuous: clip the sampled vector to the
                # env's [-1, 1] box (we store the UNCLIPPED action for the log-prob)
                step_action = np.clip(actions[i], -1.0, 1.0) if self.continuous else int(actions[i])
                next_obs_i, reward, terminated, truncated, _ = env.step(step_action)
                ep_return[i] += reward  # RAW reward for logging (eval/train returns stay in real scale)

                # reward normalization: divide by a running std of the discounted return
                # (gym/CleanRL convention) — pulls the value/return scale to ~O(1) so the
                # critic targets, GAE, and the 0.2 value-clip all live on a sane scale.
                if self.normalize_reward:
                    self._ret_acc[i] = self.gamma * self._ret_acc[i] + reward
                    self.return_rms.update(np.array([self._ret_acc[i]]))
                    reward = float(np.clip(reward / np.sqrt(self.return_rms.var + 1e-8), -10.0, 10.0))
                rewards_buf[t, i] = reward

                # Per-env truncation bootstrap (same fix as single-env): a time-limit
                # truncation is NOT a real terminal, so fold gamma * V(s_T) into the
                # reward before resetting; GAE still masks the boundary via dones.
                if truncated and not terminated:
                    with torch.no_grad():
                        boot_obs = self._norm_obs(next_obs_i)
                        rewards_buf[t, i] += self.gamma * float(self.critic(self._tensor(boot_obs)).item())

                done = terminated or truncated
                if done:
                    ep_returns.append(float(ep_return[i]))
                    ep_return[i] = 0.0
                    self._ret_acc[i] = 0.0  # reset discounted-return accumulator at episode end
                    next_obs_i, _ = env.reset()

                self._next_obs[i] = next_obs_i
                self._next_done[i] = float(done)

        with torch.no_grad():
            next_value = self.critic(self._tensor(self._norm_obs(self._next_obs))).cpu().numpy()  # [N]

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
        actions = torch.as_tensor(
            buffer["actions"],
            dtype=torch.float32 if self.continuous else torch.long,
            device=self.device,
        )  # [T, N] discrete; [T, N, action_dim] continuous
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
        actions = actions.reshape(-1, self.action_dim) if self.continuous else actions.reshape(-1)
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
