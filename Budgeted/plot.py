import numpy as np
from matplotlib import pyplot as plt

from Budgeted.c_b_thompson import ThompsonSamplingContextualBandit
from Budgeted.olr_e_greedy import EpsilonGreedyContextualBandit
from Budgeted.lin_ucb import LinUCB
from Budgeted.w_ucb import OmegaUCB

# Set parameters
num_arms = 3
num_features = 3
num_rounds =10000000
true_weights =  np.array([[0.4359949, 0.02592623, 0.54966248], [0.43532239, 0.4203678, 0.33033482], [0.20464863, 0.61927097, 0.29965467]])
context = np.random.rand(num_rounds, num_features)
epsilon = 0.1  # Exploration rate
true_cost= np.array([0.34014455, 0.65902045, 0.57622788])
budget = 1000
variance = 0.1
alpha = 0.2

#bandit = ThompsonSamplingContextualBandit(num_features, variance, num_arms, context, true_weights, true_cost, budget)
#bandit.run()
#ucb_bandit = LinUCB(n_actions=num_arms, n_features=num_features, contexts=context, true_theta=true_weights, cost=true_cost, alpha=alpha, budget=budget)
#ucb_bandit.run()
#bandit_eg = EpsilonGreedyContextualBandit(num_features, epsilon, num_arms, context, true_weights, true_cost, budget=budget)
#bandit_eg.run()
#print('muHat', bandit_eg.mu_hat)
omega_ucb = OmegaUCB(n_actions=num_arms, n_features=num_features, contexts=context, true_theta=true_weights, cost=true_cost, alpha=alpha, budget=budget)
omega_ucb.run()

# Plot des Regret
plt.subplot(2, 1, 2)
plt.plot(np.cumsum(np.array(bandit.optimal_reward) - np.array(bandit.observed_reward_history)), label="Regret Budgeted Thompson", color='red')
#plt.plot(np.cumsum(np.array(np.array(bandit_eg.optimal_reward) - np.array(bandit_eg.observed_reward_history))), label="Regret Budgeted Epsilon Greedy", color='blue')
#plt.plot(np.cumsum(ucb_bandit.optimal_reward) - np.cumsum(ucb_bandit.rewards[:len(ucb_bandit.optimal_reward)]), label="Regret Budgeted LinUCB", color='grey')
#plt.plot(np.cumsum(omega_ucb.optimal_reward) - np.cumsum(omega_ucb.rewards[:len(omega_ucb.optimal_reward)]), label="Regret Budgeted OmegaLinUCB", color='green')
plt.xlabel("Runden")
plt.ylabel("Regret")
plt.title("Regret über Zeit")
plt.legend()

plt.tight_layout()
plt.show()