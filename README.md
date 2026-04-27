This project evaluates and compares multi-step temporal-difference reinforcement learning methods, specifically n-step SARSA and SARSA(λ), using the FrozenLake-v1 environment from Gymnasium.

Environment details: 8x8 grid, is_slippery=False

Reward function: -1 for falling into terminal hole, -0.01 for each step, +1 for reaching goal

SARSA (n-step): n=3, n=10, n=100

SARSA(λ): λ= 0.9 , λ= 0.7 

Different values of n and lambda can be tested by adjusting those values in lake_multiplen and lake_lambda.

The methods are evaluated based on the following metrics:
Computation time(seconds)
Average reward - plotted on graph 
Success rate - an episode is considered successful if the reward is positive at the end of the episode 
Time to convergence - defined as reaching 95% of the average reward for 2000 episodes 

Python 3.10 or higher is needed to run the code due to the adjusted reward function.
Aligned with algorithms outlined in Sutton & Barto: Reinforcement Learning.
