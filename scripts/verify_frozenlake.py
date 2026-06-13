import gymnasium as gym
from rlfoundations.algorithms.q_learning import QLearningAgent


env = gym.make("FrozenLake-v1", is_slippery=False)

agent = QLearningAgent(
    actions_fn=lambda state: list(range(env.action_space.n)),
    alpha=0.1,
    gamma=0.99,
    epsilon=1.0,
    epsilon_decay=0.995,
    min_epsilon=0.05,
)

agent.train(env, num_episodes=5000, log_every=500)

state, _ = env.reset()
done = False

while not done:
    action = agent.best_action(state)
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated