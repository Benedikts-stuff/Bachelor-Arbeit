import numpy as np
from matplotlib import pyplot as plt

from Budgeted.c_b_thompson import ThompsonSamplingContextualBandit
from Budgeted.olr_e_greedy import EpsilonGreedyContextualBandit
from Budgeted.lin_ucb import LinUCB

# Set parameters
num_arms = 3
num_features = 3
num_rounds =100000
true_weights = np.array([[0.5, 0.1, 0.2], [0.1, 0.5, 0.2], [0.2, 0.1, 0.5]])
context = np.random.rand(num_rounds, num_features)
epsilon = 0.1  # Exploration rate
true_cost= np.array([0.8, 1, 0.6])
budget = 1500
variance = 0.2
alpha = 0.2

bandit = ThompsonSamplingContextualBandit(num_features, variance, num_arms, num_rounds, context, true_weights, true_cost, budget)
bandit.run()
ucb_bandit = LinUCB(n_actions=num_arms, n_features=num_features, contexts=context, true_theta=true_weights, cost=true_cost, alpha=alpha, budget=budget)
ucb_bandit.run()
bandit_eg = EpsilonGreedyContextualBandit(num_features, epsilon, num_rounds, num_arms, context, true_weights, true_cost, budget)
bandit_eg.run()

# Plot des Regret
plt.subplot(2, 1, 2)
plt.plot(np.cumsum(np.array(bandit.optimal_reward) - np.array(bandit.observed_reward_history)), label="Regret", color='red')
plt.plot(np.cumsum(np.array(np.array(bandit_eg.optimal_reward) - np.array(bandit_eg.observed_reward_history))), label="Regret", color='blue')
plt.plot(np.cumsum(ucb_bandit.optimal_reward) - np.cumsum(ucb_bandit.rewards[:len(ucb_bandit.optimal_reward)]), label="Regret", color='grey')
plt.xlabel("Runden")
plt.ylabel("Regret")
plt.title("Regret über Zeit")
plt.legend()

plt.tight_layout()
plt.show()