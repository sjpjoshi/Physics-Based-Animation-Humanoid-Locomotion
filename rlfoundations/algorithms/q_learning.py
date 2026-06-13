"""Tabular Q-learning agent (Step 1).

Promoted from scripts/ into the package so it's reusable. The agent is
environment-agnostic over any *hashable* state, which is exactly why the CartPole
runner can put a discretizer in front of it without changing anything here.
"""
from collections import defaultdict
import random
from typing import Any, Callable, Dict, Hashable, Iterable, List, Tuple


State = Hashable
Action = Any
QTable = Dict[Tuple[State, Action], float]


class QLearningAgent:
    def __init__(
        self,
        actions_fn: Callable[[State], List[Action]],
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        epsilon_decay: float = 1.0,
        min_epsilon: float = 0.01,
    ):
        """
        Args:
            actions_fn:
                Function that returns valid actions for a given state.

            alpha:
                Learning rate.

            gamma:
                Discount factor.

            epsilon:
                Probability of taking a random exploratory action.

            epsilon_decay:
                Multiplicative decay applied to epsilon after each episode.

            min_epsilon:
                Lower bound on epsilon.
        """
        self.actions_fn = actions_fn
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.q: QTable = defaultdict(float)

    def get_q(self, state: State, action: Action) -> float:
        return self.q[(state, action)]

    def choose_action(self, state: State) -> Action:
        """
        Epsilon-greedy action selection.

        With probability epsilon, explore.
        Otherwise, exploit the highest-Q action.
        """
        actions = self.actions_fn(state)

        if not actions:
            raise ValueError(f"No available actions for state: {state}")

        if random.random() < self.epsilon:
            return random.choice(actions)

        return self.best_action(state)

    def best_action(self, state: State) -> Action:
        """
        Return the action with the highest current Q-value.
        Break ties randomly.
        """
        actions = self.actions_fn(state)

        if not actions:
            raise ValueError(f"No available actions for state: {state}")

        max_q = max(self.get_q(state, action) for action in actions)
        best_actions = [
            action for action in actions
            if self.get_q(state, action) == max_q
        ]

        return random.choice(best_actions)

    def update(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        done: bool,
    ) -> float:
        """
        Q-learning update:

            Q(s,a) <- Q(s,a) + alpha * [target - Q(s,a)]

        where:

            target = r                         if terminal
            target = r + gamma * max_a Q(s',a) otherwise

        Returns:
            TD error, useful for debugging.
        """
        old_q = self.get_q(state, action)

        if done:
            target = reward
        else:
            next_actions = self.actions_fn(next_state)
            if next_actions:
                best_next_q = max(self.get_q(next_state, a) for a in next_actions)
            else:
                best_next_q = 0.0

            target = reward + self.gamma * best_next_q

        td_error = target - old_q
        self.q[(state, action)] = old_q + self.alpha * td_error

        return td_error

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def train_episode(self, env, max_steps: int = 10_000) -> float:
        """
        Train for one episode.

        Assumes Gymnasium-style API:
            state, info = env.reset()
            next_state, reward, terminated, truncated, info = env.step(action)

        Returns:
            Total episode reward.
        """
        state, _ = env.reset()
        total_reward = 0.0

        for _ in range(max_steps):
            action = self.choose_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            self.update(state, action, reward, next_state, done)

            total_reward += reward
            state = next_state

            if done:
                break

        self.decay_epsilon()
        return total_reward

    def train(
        self,
        env,
        num_episodes: int,
        max_steps_per_episode: int = 10_000,
        log_every: int = 100,
    ) -> List[float]:
        """
        Train over many episodes.

        Returns:
            List of episode rewards.
        """
        rewards = []

        for episode in range(1, num_episodes + 1):
            episode_reward = self.train_episode(env, max_steps_per_episode)
            rewards.append(episode_reward)

            if log_every and episode % log_every == 0:
                avg_reward = sum(rewards[-log_every:]) / log_every
                print(
                    f"Episode {episode:5d} | "
                    f"Avg Reward: {avg_reward:8.3f} | "
                    f"Epsilon: {self.epsilon:.4f}"
                )

        return rewards
