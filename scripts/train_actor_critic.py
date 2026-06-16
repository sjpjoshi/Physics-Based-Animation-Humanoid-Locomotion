"""Step 3 runner — actor-critic on CartPole.

Same end-to-end plumbing as Step 2 (build env -> agent -> one full-episode update
at a time -> periodic greedy eval -> log to Weights & Biases), now driving an actor
*and* a critic. The three pieces you implement live in
rlfoundations/algorithms/actor_critic.py: `compute_advantages`, `actor_loss`, and
`critic_loss`. Once all three are in, run from the repo root:

    python scripts/train_actor_critic.py
"""
from __future__ import annotations

import numpy as np

from rlfoundations.algorithms.actor_critic import ActorCriticAgent
from rlfoundations.envs import make_env
from rlfoundations.utils.logging import Logger
from rlfoundations.utils.seeding import set_seed

# --- knobs (tune these) ---------------------------------------------------
SEED = 0
NUM_EPISODES = 1500
HIDDEN_SIZES = (128,)
ACTOR_LR = 1e-3        # policy step; same stable zone as REINFORCE
CRITIC_LR = 1e-3       # value-net step; can run a touch higher (e.g. 3e-3) so the baseline keeps up
GAMMA = 0.99
NORMALIZE_ADVANTAGES = True   # standardize A_t before the actor loss; same variance win as Step 2's NORMALIZE_RETURNS

EVAL_EVERY = 50               # run a greedy-eval block every N training episodes
EVAL_EPISODES = 20            # episodes per periodic eval block
FINAL_EVAL_EPISODES = 100     # the "solved?" measure at the end
SOLVED_RETURN = 195.0
LOG_EVERY = 50                # console-print cadence
# --------------------------------------------------------------------------


def evaluate(agent: ActorCriticAgent, env, n_episodes: int) -> list[float]:
    """Greedy (argmax, no sampling) evaluation; returns per-episode returns."""
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(agent.act_greedy(obs))
            total += reward
            done = terminated or truncated
        returns.append(total)
    return returns


def main() -> None:
    set_seed(SEED)

    env = make_env("CartPole-v1", seed=SEED)
    eval_env = make_env("CartPole-v1", seed=SEED + 1000)  # held-out seed for eval

    agent = ActorCriticAgent(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        hidden_sizes=HIDDEN_SIZES,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        gamma=GAMMA,
        normalize_advantages=NORMALIZE_ADVANTAGES,
    )

    config = {
        "algo": "actor_critic",
        "num_episodes": NUM_EPISODES,
        "hidden_sizes": list(HIDDEN_SIZES),
        "actor_lr": ACTOR_LR,
        "critic_lr": CRITIC_LR,
        "gamma": GAMMA,
        "normalize_advantages": NORMALIZE_ADVANTAGES,
    }

    recent: list[float] = []
    with Logger("rlfoundations", config, run_name="actor-critic-cartpole") as logger:
        for episode in range(1, NUM_EPISODES + 1):
            ep_return = agent.train_episode(env)
            recent.append(ep_return)

            metrics = {"train_return": ep_return}
            if episode % EVAL_EVERY == 0:
                eval_mean = float(np.mean(evaluate(agent, eval_env, EVAL_EPISODES)))
                metrics["eval_mean_return"] = eval_mean
                print(f"           eval @ ep {episode:5d}: mean {eval_mean:7.1f} over {EVAL_EPISODES} eps")
            logger.log(metrics, step=episode)

            if episode % LOG_EVERY == 0:
                avg = sum(recent[-LOG_EVERY:]) / LOG_EVERY
                print(f"Episode {episode:5d} | avg train return (last {LOG_EVERY}): {avg:8.2f}")

        final_mean = float(np.mean(evaluate(agent, eval_env, FINAL_EVAL_EPISODES)))
        logger.run.summary["final_eval_mean_return"] = final_mean

    env.close()
    eval_env.close()

    print(
        f"\nFinal greedy eval: mean return {final_mean:.1f} over "
        f"{FINAL_EVAL_EPISODES} episodes (target {SOLVED_RETURN:.0f})."
    )
    if final_mean >= SOLVED_RETURN:
        print("SOLVED. Step 3 done.")
    else:
        print("Not solved yet. Tune ACTOR_LR / CRITIC_LR / NUM_EPISODES / NORMALIZE_ADVANTAGES.")


if __name__ == "__main__":
    main()
