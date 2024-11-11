import numpy as np
import matplotlib.pyplot as plt
#from experiment import BanditLogger




class EpsilonGreedyContextualBandit:
    def __init__(self, d, epsilon,  n_arms, contexts, true_weights, true_cost, budget, logger, repetition, seed):
        """
        d: Dimension der Kontextvektoren
        epsilon: Wahrscheinlichkeit für Exploration
        """
        np.random.seed(42)
        self.repetition = repetition
        self.logger = logger
        self.n_features = d
        self.epsilon = epsilon
        self.n_arms = n_arms
        self.B = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])
        self.f = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        self.mu_hat = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        self.budget = budget
        self.og_budget = budget
        self.costs = true_cost
        self.empirical_cost_means = np.random.rand(self.n_arms)
        self.arm_counts = np.zeros(self.n_arms)
        self.max_cost = np.max(self.costs)
        self.i = 0
        self.gamma = 0.00000001
        self.cum = np.zeros(self.n_arms)
        self.contexts = contexts
        self.true_weights = true_weights
        self.observed_reward_history = []
        self.actual_reward_history = self.contexts.dot(self.true_weights.T)
        self.optimal_reward = []
        self.summed_regret = 0

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
        count = 0
        while self.budget > self.max_cost:
            context = self.contexts[i]
            chosen_arm = self.select_arm(context)
            self.update(self.actual_reward_history[i, chosen_arm], chosen_arm, context)
            observed_reward = self.actual_reward_history[i, chosen_arm] / self.costs[chosen_arm]
            self.observed_reward_history.append(observed_reward)

            optimal = np.max(self.actual_reward_history[i] / self.costs)
            opt_arm = np.argmax(self.actual_reward_history[i] / self.costs)
            self.optimal_reward.append(optimal - observed_reward)

            if opt_arm != chosen_arm:
                count+=1
            self.summed_regret += optimal - observed_reward

            self.logger.track_rep(self.repetition)
            self.logger.track_approach(0)
            self.logger.track_round(i)
            self.logger.track_regret(self.summed_regret)
            self.logger.track_normalized_budget((self.og_budget - self.budget)/ self.og_budget)
            self.logger.track_spent_budget(self.og_budget - self.budget)
            self.logger.finalize_round()
            i += 1





# Set parameters
num_arms = 3
num_features = 3
num_rounds =100000
true_weights = np.array([[0.5, 0.1, 0.2], [0.1, 0.5, 0.2], [0.2, 0.1, 0.5]])
context = np.random.rand(num_rounds, num_features)
epsilon = 0.1  # Exploration rate
true_cost= np.array([0.8, 1, 0.6])
budget = 1500
#bandit_eg = EpsilonGreedyContextualBandit(num_features, epsilon, num_arms, context, true_weights, true_cost, budget)
#bandit_eg.run()
#print(bandit_eg.empirical_cost_means)
#print(bandit_eg.mu_hat)

# Plot comparison
#optimal_reward = bandit_eg.optimal_reward
#regret_eg = np.array(optimal_reward) - np.array(bandit_eg.observed_reward_history)



# Kumulative Belohnung

