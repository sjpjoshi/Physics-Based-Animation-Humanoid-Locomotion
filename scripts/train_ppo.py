from __future__ import annotations

import numpy as np

from rlfoundations.algorithms.ppo import PPOAgent
from rlfoundations.envs import make_env, make_vector_env
from rlfoundations.utils.logging import Logger
from rlfoundations.utils.seeding import set_seed

# --- knobs (tune these) ---------------------------------------------------
SEED = 0
TOTAL_TIMESTEPS = 250_000
NUM_ENVS = 4                  # parallel CartPoles (hand-rolled vector); batch = NUM_STEPS * NUM_ENVS
NUM_STEPS = 128               # rollout horizon PER env per iteration (128 * 4 = 512 batch)
HIDDEN_SIZES = (64, 64)       # CleanRL CartPole arch (Tanh trunk via networks.mlp)
LR = 2.5e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
UPDATE_EPOCHS = 4
NUM_MINIBATCHES = 4           # 512 / 4 = 128 per minibatch
NORMALIZE_ADVANTAGES = True
ANNEAL_LR = True              # linearly decay LR -> 0 over training (kills late overshoot)
CLIP_VLOSS = True             # clip the value loss to a trust region (critic analogue of policy clip)
ORTHOGONAL_INIT = True        # PPO-standard orthogonal weight init

EVAL_EVERY_ITERS = 10         # greedy-eval block every N iterations
EVAL_EPISODES = 20
FINAL_EVAL_EPISODES = 100
SOLVED_RETURN = 195.0
# --------------------------------------------------------------------------


def evaluate(agent: PPOAgent, env, n_episodes: int) -> list[float]:
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

    envs = make_vector_env("CartPole-v1", NUM_ENVS, seed=SEED)
    eval_env = make_env("CartPole-v1", seed=SEED + 1000)

    agent = PPOAgent(
        obs_dim=envs[0].observation_space.shape[0],
        n_actions=envs[0].action_space.n,
        hidden_sizes=HIDDEN_SIZES,
        lr=LR,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_coef=CLIP_COEF,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        update_epochs=UPDATE_EPOCHS,
        num_minibatches=NUM_MINIBATCHES,
        normalize_advantages=NORMALIZE_ADVANTAGES,
        clip_vloss=CLIP_VLOSS,
        orthogonal_init=ORTHOGONAL_INIT,
    )

    num_iterations = TOTAL_TIMESTEPS // (NUM_STEPS * NUM_ENVS)

    config = {
        "algo": "ppo",
        "total_timesteps": TOTAL_TIMESTEPS,
        "num_envs": NUM_ENVS,
        "num_steps": NUM_STEPS,
        "batch_size": NUM_STEPS * NUM_ENVS,
        "num_iterations": num_iterations,
        "hidden_sizes": list(HIDDEN_SIZES),
        "lr": LR,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "clip_coef": CLIP_COEF,
        "ent_coef": ENT_COEF,
        "vf_coef": VF_COEF,
        "update_epochs": UPDATE_EPOCHS,
        "num_minibatches": NUM_MINIBATCHES,
        "normalize_advantages": NORMALIZE_ADVANTAGES,
        "anneal_lr": ANNEAL_LR,
        "clip_vloss": CLIP_VLOSS,
        "orthogonal_init": ORTHOGONAL_INIT,
    }

    with Logger("rlfoundations", config, run_name="ppo-cartpole") as logger:
        for iteration in range(1, num_iterations + 1):
            if ANNEAL_LR:
                frac = 1.0 - (iteration - 1) / num_iterations
                for pg in agent.optimizer.param_groups:
                    pg["lr"] = frac * LR

            buffer, ep_returns = agent.collect_rollout(envs, NUM_STEPS)
            stats = agent.update(buffer)
            global_step = iteration * NUM_STEPS * NUM_ENVS

            metrics = dict(stats)
            metrics["lr"] = agent.optimizer.param_groups[0]["lr"]
            if ep_returns:
                metrics["train_return"] = float(np.mean(ep_returns))

            if iteration % EVAL_EVERY_ITERS == 0:
                eval_mean = float(np.mean(evaluate(agent, eval_env, EVAL_EPISODES)))
                metrics["eval_mean_return"] = eval_mean
                train_str = f"{metrics['train_return']:7.1f}" if "train_return" in metrics else "   n/a "
                print(
                    f"iter {iteration:4d} | step {global_step:7d} | "
                    f"train {train_str} | eval {eval_mean:7.1f} over {EVAL_EPISODES} eps"
                )

            logger.log(metrics, step=global_step)

        final_mean = float(np.mean(evaluate(agent, eval_env, FINAL_EVAL_EPISODES)))
        logger.run.summary["final_eval_mean_return"] = final_mean

    for env in envs:
        env.close()
    eval_env.close()

    print(
        f"\nFinal greedy eval: mean return {final_mean:.1f} over "
        f"{FINAL_EVAL_EPISODES} episodes (target {SOLVED_RETURN:.0f})."
    )
    if final_mean >= SOLVED_RETURN:
        print("SOLVED. Step 4 done — PPO + GAE.")
    else:
        print("Not solved yet. Tune LR / NUM_STEPS / UPDATE_EPOCHS / ENT_COEF / CLIP_COEF.")


if __name__ == "__main__":
    main()
