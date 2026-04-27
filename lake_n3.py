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
episodes = 100000
n = 3

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

    states = [state]
    actions = [action]
    rewards = [0]

    done_time = float("inf")
    t = 0
    total_reward = 0

    while True:
        if t < done_time:
            next_state, reward, terminated, truncated, _ = env.step(actions[t])
            total_reward += reward

            if terminated and reward > 0:
                success_count += 1

            rewards.append(reward)
            states.append(next_state)

            if terminated or truncated:
                done_time = t + 1
            else:
                next_action = epsilon_greedy(Q, next_state, epsilon)
                actions.append(next_action)

        update_time = t - n + 1

        if update_time >= 0:
            target = 0.0

            upper = min(update_time + n, done_time)
            for i in range(update_time + 1, int(upper) + 1):
                target += (gamma ** (i - update_time - 1)) * rewards[i]

            if update_time + n < done_time:
                target += (gamma ** n) * Q[states[update_time + n], actions[update_time + n]]

            s_update_time = states[update_time]
            a_update_time = actions[update_time]
            Q[s_update_time, a_update_time] += alpha * (target - Q[s_update_time, a_update_time])

        if update_time == done_time - 1:
            break

        t += 1

    episode_rewards.append(total_reward)

env.close()

end_time = time.perf_counter()
total_time = end_time - start_time

window = 10000
moving_avg = []

for i in range(len(episode_rewards)):
    start = max(0, i - window + 1)
    moving_avg.append(np.mean(episode_rewards[start:i+1]))

t_conv = time_to_convergence(moving_avg)
success_rate = success_count / episodes

print("Training finished.")
print("Average reward:", np.mean(episode_rewards))
print("Success rate:", success_rate)
print("Total computation time (seconds):", total_time)
print("Time to convergence:", t_conv)

plt.plot(moving_avg)
plt.xlabel("Episode")
plt.ylabel("Average Reward (last 10000 episodes)")
plt.title("100-step SARSA on FrozenLake")
plt.show()
