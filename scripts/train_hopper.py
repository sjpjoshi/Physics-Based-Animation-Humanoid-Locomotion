"""Phase-1 capstone — PPO on Hopper-v5 (continuous control).

The same PPO machine as CartPole, now driving a continuous (diagonal-Gaussian)
policy. The ONLY new core was GaussianPolicy.distribution in rlfoundations/networks.py
— compute_gae, ppo_policy_loss, the clipped objective, GAE, and every hardening item
carry over untouched. That reuse is the whole point of Hopper: it proves the PPO
isn't CartPole-specific.

Continuous-PPO hyper-params follow CleanRL's ppo_continuous_action.py.

STAGING NOTE: this first cut runs RAW Hopper with NO observation/reward normalization,
to keep the focus on the continuous policy. Hopper will learn to hop but won't reach
top scores (~2500+) without normalization — that's the planned Hopper hardening pass
(the MuJoCo analogue of the CartPole hardening we just did).

    python scripts/train_hopper.py
"""
from __future__ import annotations

import os

import numpy as np
import torch

from rlfoundations.algorithms.ppo import PPOAgent
from rlfoundations.envs import make_env, make_vector_env
from rlfoundations.utils.logging import Logger
from rlfoundations.utils.seeding import set_seed

# --- knobs (CleanRL ppo_continuous_action defaults) -----------------------
SEED = 0
ENV_ID = "Hopper-v5"
TOTAL_TIMESTEPS = int(os.environ.get("HOPPER_TIMESTEPS", 1_000_000))  # env override for quick runs
NUM_ENVS = 1
NUM_STEPS = 2048             # long rollouts, standard for continuous control
HIDDEN_SIZES = (64, 64)
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
ENT_COEF = 0.0              # continuous PPO usually runs no entropy bonus
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
UPDATE_EPOCHS = 10
NUM_MINIBATCHES = 32        # 2048 / 32 = 64 per minibatch
NORMALIZE_ADVANTAGES = True
ANNEAL_LR = True
CLIP_VLOSS = True
ORTHOGONAL_INIT = True
NORMALIZE_OBS = True         # running mean/std on observations — high-impact for MuJoCo
NORMALIZE_REWARD = True      # running std of the discounted return — pulls returns to ~O(1)

EVAL_EVERY_ITERS = 10
EVAL_EPISODES = 10
FINAL_EVAL_EPISODES = 20
# Hopper reference returns: random ~15, learning ~1000, strong ~2500+.
# --------------------------------------------------------------------------


def evaluate(agent: PPOAgent, env, n_episodes: int) -> list[float]:
    """Greedy (mean-action) evaluation; returns per-episode returns."""
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

    envs = make_vector_env(ENV_ID, NUM_ENVS, seed=SEED)
    eval_env = make_env(ENV_ID, seed=SEED + 1000)

    obs_dim = envs[0].observation_space.shape[0]
    action_dim = envs[0].action_space.shape[0]

    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        continuous=True,
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
        normalize_obs=NORMALIZE_OBS,
        normalize_reward=NORMALIZE_REWARD,
    )

    num_iterations = TOTAL_TIMESTEPS // (NUM_STEPS * NUM_ENVS)

    config = {
        "algo": "ppo_continuous",
        "env_id": ENV_ID,
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
        "anneal_lr": ANNEAL_LR,
        "clip_vloss": CLIP_VLOSS,
        "orthogonal_init": ORTHOGONAL_INIT,
        "normalize_obs": NORMALIZE_OBS,
        "normalize_reward": NORMALIZE_REWARD,
    }

    with Logger("rlfoundations", config, run_name="ppo-hopper") as logger:
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
                train_str = f"{metrics['train_return']:8.1f}" if "train_return" in metrics else "    n/a "
                print(
                    f"iter {iteration:4d} | step {global_step:8d} | "
                    f"train {train_str} | eval {eval_mean:8.1f} over {EVAL_EPISODES} eps"
                )

            logger.log(metrics, step=global_step)

        final_mean = float(np.mean(evaluate(agent, eval_env, FINAL_EVAL_EPISODES)))
        logger.run.summary["final_eval_mean_return"] = final_mean

    for env in envs:
        env.close()
    eval_env.close()

    print(
        f"\nFinal greedy eval: mean return {final_mean:.1f} over {FINAL_EVAL_EPISODES} episodes."
    )
    print("Hopper reference: random ~15, learning ~1000, strong ~2500+ (top scores need normalization).")

    # save a checkpoint for rendering / reuse: actor weights + the obs normalizer
    # (the policy was trained on normalized obs, so eval/render must normalize too)
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(
        {
            "actor": agent.actor.state_dict(),
            "obs_rms_mean": agent.obs_rms.mean if agent.obs_rms is not None else None,
            "obs_rms_var": agent.obs_rms.var if agent.obs_rms is not None else None,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "hidden_sizes": list(HIDDEN_SIZES),
            "final_eval": final_mean,
        },
        "checkpoints/hopper_ppo.pt",
    )
    print("saved checkpoint -> checkpoints/hopper_ppo.pt")


if __name__ == "__main__":
    main()
