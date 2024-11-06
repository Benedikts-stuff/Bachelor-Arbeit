import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)

class EpsilonGreedyContextualBandit:
    def __init__(self, d, epsilon,num_rounds,  n_arms, contexts, true_weights, true_cost, budget):
        """
        d: Dimension der Kontextvektoren
        epsilon: Wahrscheinlichkeit für Exploration
        """
        self.n_features = d
        self.epsilon = epsilon
        self.n_arms = n_arms
        self.B = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])
        self.f = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        self.mu_hat = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        self.budget = budget
        self.costs = true_cost
        self.empirical_cost_means = np.random.rand(self.n_arms)
        self.arm_counts = np.zeros(self.n_arms)
        self.max_cost = np.max(self.costs)
        self.i = 0
        self.num_rounds = num_rounds
        self.gamma = 0.00000001
        self.cum = np.zeros(self.n_arms)
        self.contexts = contexts
        self.true_weights = true_weights
        self.observed_reward_history = []
        self.actual_reward_history = self.contexts.dot(self.true_weights.T)
        self.optimal_reward = []

    def select_arm(self, context):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_arms)  # Exploration
        else:
            expected_rewards = np.array([np.dot(self.mu_hat[i], context) for i in range(self.n_arms)])
            return np.argmax(expected_rewards / (self.empirical_cost_means + self.gamma))  # Exploitation

    def update(self, reward, chosen_arm, context):
        self.B[chosen_arm] += np.outer(context, context)
        self.f[chosen_arm] += reward * context
        self.mu_hat[chosen_arm] = np.linalg.inv(self.B[chosen_arm]).dot(self.f[chosen_arm])

        self.cum[chosen_arm] += np.random.binomial(1, self.costs[chosen_arm])
        self.empirical_cost_means[chosen_arm] =  self.cum[chosen_arm] / (self.arm_counts[chosen_arm] +1)

        self.arm_counts[chosen_arm] += 1

        self.budget = self.budget - self.costs[chosen_arm]

    def run(self):
        i = 0
        while self.budget > self.max_cost:
            context = self.contexts[i]
            chosen_arm = self.select_arm(context)
            self.update(self.actual_reward_history[i, chosen_arm], chosen_arm, context)
            self.observed_reward_history.append(self.actual_reward_history[i, chosen_arm] / self.costs[chosen_arm])

            optimal = np.max(self.actual_reward_history[i] / self.costs)
            self.optimal_reward.append(optimal)
            i +=1



# Set parameters
num_arms = 3
num_features = 3
num_rounds =100000
true_weights = np.array([[0.5, 0.1, 0.2], [0.1, 0.5, 0.2], [0.2, 0.1, 0.5]])
context = np.random.rand(num_rounds, num_features)
epsilon = 0.1  # Exploration rate
true_cost= np.array([0.8, 1, 0.6])
budget = 1500
bandit_eg = EpsilonGreedyContextualBandit(num_features, epsilon, num_rounds, num_arms, context, true_weights, true_cost, budget)
bandit_eg.run()
print(bandit_eg.empirical_cost_means)
print(bandit_eg.mu_hat)

# Plot comparison
optimal_reward = bandit_eg.optimal_reward
regret_eg = np.array(optimal_reward) - np.array(bandit_eg.observed_reward_history)



# Kumulative Belohnung

