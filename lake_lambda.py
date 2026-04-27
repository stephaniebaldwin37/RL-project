import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False, reward_schedule=(1, -1, -0.01))

num_states = env.observation_space.n
num_actions = env.action_space.n

Q = np.zeros((num_states, num_actions))

alpha = 0.1
gamma = 0.99
epsilon = 0.1
lam = 0.9
episodes = 100000

def epsilon_greedy(Q, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(num_actions)
    return np.argmax(Q[state])

def time_to_convergence(moving_avg, sustain=2000, threshold_ratio=0.95, tail=5000):
    final_perf = np.mean(moving_avg[-tail:])
    threshold = threshold_ratio * final_perf

    for i in range(len(moving_avg) - sustain):
        if np.all(np.array(moving_avg[i:i+sustain]) >= threshold):
            return i
    return None

episode_rewards = []
success_count = 0

start_time = time.perf_counter()

for episode in range(episodes):
    state, _ = env.reset()
    action = epsilon_greedy(Q, state, epsilon)

    E = np.zeros((num_states, num_actions))
    total_reward = 0

    while True:
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated and reward > 0:
            success_count += 1

        if terminated or truncated:
            td_target = reward
            td_error = td_target - Q[state, action]
        else:
            next_action = epsilon_greedy(Q, next_state, epsilon)
            td_target = reward + gamma * Q[next_state, next_action]
            td_error = td_target - Q[state, action]

        E[state, action] += 1

        Q += alpha * td_error * E

        E *= gamma * lam

        if terminated or truncated:
            break

        state = next_state
        action = next_action

    episode_rewards.append(total_reward)

end_time = time.perf_counter()
total_time = end_time - start_time

window = 10000
moving_avg_lambda = []

for i in range(len(episode_rewards)):
    start = max(0, i - window + 1)
    moving_avg_lambda.append(np.mean(episode_rewards[start:i+1]))

t_conv = time_to_convergence(moving_avg_lambda)
success_rate = success_count / episodes

print("Training finished.")
print("Average reward:", np.mean(episode_rewards))
print("Success rate:", success_rate)
print("Total computation time:", total_time)
print("Time to convergence:", t_conv)

env.close()

plt.plot(moving_avg_lambda)
plt.xlabel("Episode")
plt.ylabel("Average Reward (last 1000)")
plt.title("SARSA(lambda = 0.9) on FrozenLake")
plt.show()
