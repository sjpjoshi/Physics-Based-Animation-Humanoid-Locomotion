"""Step 1 runner — tabular Q-learning on CartPole.

Plumbing is wired end to end: make env -> discretize observations -> train the
Q-learning agent -> greedy eval over 100 episodes -> log to Weights & Biases.

The ONE piece you implement is ``Discretizer.__call__`` in
rlfoundations/discretization.py. Once that's in, run from the repo root:

    python scripts/train_q_learning.py
"""
from __future__ import annotations

import numpy as np

from rlfoundations.algorithms.q_learning import QLearningAgent
from rlfoundations.discretization import Discretizer, DiscretizedObservation
from rlfoundations.envs import make_env
from rlfoundations.utils.logging import Logger
from rlfoundations.utils.seeding import set_seed

# --- knobs (tune these) ---------------------------------------------------
SEED = 0
N_BINS = 12
NUM_EPISODES = 5000          # tabular CartPole often needs more — bump if it plateaus
ALPHA = 0.1
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_DECAY = 0.999        # per-episode multiplicative decay
MIN_EPSILON = 0.05
EVAL_EPISODES = 100
SOLVED_RETURN = 195.0

# CartPole obs = [cart pos, cart velocity, pole angle, pole angular velocity].
# pos/angle use the episode's termination limits; the velocity dims are unbounded,
# so we clip them to a hand-picked range (tune these too).
OBS_LOW = [-2.4, -3.0, -0.21, -3.0]
OBS_HIGH = [2.4, 3.0, 0.21, 3.0]
# --------------------------------------------------------------------------


def evaluate(agent: QLearningAgent, env, n_episodes: int) -> list[float]:
    """Greedy (no-exploration) evaluation; returns per-episode returns."""
    returns = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            action = agent.best_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        returns.append(total)
    return returns


def main() -> None:
    set_seed(SEED)

    discretizer = Discretizer(OBS_LOW, OBS_HIGH, N_BINS)
    env = DiscretizedObservation(make_env("CartPole-v1", seed=SEED), discretizer)

    agent = QLearningAgent(
        actions_fn=lambda state: list(range(env.action_space.n)),
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON_START,
        epsilon_decay=EPSILON_DECAY,
        min_epsilon=MIN_EPSILON,
    )

    config = {
        "n_bins": N_BINS,
        "num_episodes": NUM_EPISODES,
        "alpha": ALPHA,
        "gamma": GAMMA,
        "epsilon_decay": EPSILON_DECAY,
        "min_epsilon": MIN_EPSILON,
    }

    with Logger("rlfoundations", config, run_name="q-learning-cartpole") as logger:
        train_returns = agent.train(env, num_episodes=NUM_EPISODES, log_every=100)
        for i, r in enumerate(train_returns):
            logger.log({"train_return": r}, step=i)

        eval_returns = evaluate(agent, env, EVAL_EPISODES)
        mean_eval = float(np.mean(eval_returns))
        logger.log({"eval_mean_return": mean_eval}, step=len(train_returns))

    env.close()

    print(
        f"\nGreedy eval: mean return {mean_eval:.1f} over {EVAL_EPISODES} episodes "
        f"(target {SOLVED_RETURN:.0f})."
    )
    if mean_eval >= SOLVED_RETURN:
        print("SOLVED. Step 1 done.")
    else:
        print("Not solved yet. Tune N_BINS / EPSILON_DECAY / NUM_EPISODES / clip ranges.")


if __name__ == "__main__":
    main()
