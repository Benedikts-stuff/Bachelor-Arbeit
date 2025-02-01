import numpy as np
import matplotlib.pyplot as plt





class EpsilonGreedyContextualBandit2:
    def __init__(self, d, epsilon,  n_arms, contexts, true_weights, true_cost, budget, logger, repetition, seed, cost_kind,alpha = 85):
        """
        d: Dimension der Kontextvektoren
        epsilon: Wahrscheinlichkeit für Exploration
        """
        np.random.seed(seed)
        self.repetition = repetition
        self.logger = logger
        self.n_features = d
        self.epsilon = 4
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
        self.cost_kind = cost_kind

        self.alpha = alpha

    def select_arm(self, context):
        #epsilon = min(1, self.alpha * (self.budget/self.og_budget)**2 * 1/(self.og_budget - self.budget +1))
        epsilon = min(1.0, 1 / 1000 * self.alpha * self.budget / (self.i + 1))
        #epsilon = min(1, self.alpha * 1/(self.og_budget - self.budget+1))
        #epsilon = min(1, self.alpha * self.budget/(self.i +1)**2)
        #epsilon = min(1, self.alpha/(self.og_budget - self.budget+1)**2)
        #epsilon =  min(1, self.alpha * np.exp(-np.exp((self.og_budget/self.budget))))

        if np.random.rand() < epsilon:
            return np.random.choice(self.n_arms)  # Exploration
        else:
            expected_rewards = np.array([np.dot(self.mu_hat[i], context) for i in range(self.n_arms)])
            if self.cost_kind == 'bernoulli':
                cost = self.empirical_cost_means
            else:
                cost = self.costs #np.random.normal(self.cost[i], 0.0001)
            return np.argmax(expected_rewards / (cost + self.gamma))  # Exploitation

    def update(self, reward, chosen_arm, context):
        self.B[chosen_arm] += np.outer(context, context)
        self.f[chosen_arm] += reward * context
        self.mu_hat[chosen_arm] = np.linalg.inv(self.B[chosen_arm]).dot(self.f[chosen_arm])

        costs = np.random.binomial(1, self.costs[chosen_arm])
        self.cum[chosen_arm] += costs
        self.empirical_cost_means[chosen_arm] =  (self.cum[chosen_arm])/ (self.arm_counts[chosen_arm] +1)

        self.arm_counts[chosen_arm] += 1

        if self.cost_kind =='bernoulli':
            self.budget = self.budget - costs
        else:
            self.budget = self.budget - self.costs[chosen_arm]

    def run(self):
        self.i = 0
        count = 0
        while self.budget > self.max_cost:
            context = self.contexts[self.i]
            chosen_arm = self.select_arm(context)
            self.update(self.actual_reward_history[self.i, chosen_arm], chosen_arm, context)
            observed_reward = self.actual_reward_history[self.i, chosen_arm] / self.costs[chosen_arm]
            self.observed_reward_history.append(observed_reward)

            optimal = np.max(self.actual_reward_history[self.i] / self.costs)
            opt_arm = np.argmax(self.actual_reward_history[self.i] / self.costs)
            self.optimal_reward.append(optimal - observed_reward)

            self.summed_regret += optimal - observed_reward

            self.logger.track_rep(self.repetition)
            self.logger.track_approach(0)
            self.logger.track_round(self.i)
            self.logger.track_regret(self.summed_regret)
            self.logger.track_normalized_budget((self.og_budget - self.budget)/ self.og_budget)
            self.logger.track_spent_budget(self.og_budget - self.budget)
            self.logger.finalize_round()
            self.i += 1





# Set parameters
num_arms = 3
num_features = 3
num_rounds =10000
true_weights = np.array([[0.5, 0.1, 0.2], [0.1, 0.5, 0.2], [0.2, 0.1, 0.5]])
context = np.random.rand(num_rounds, num_features)
epsilon = 0.2  # Exploration rate
true_cost= np.array([1, 1, 1])
budget = 1500
#bandit_eg = EpsilonGreedyContextualBandit(num_features, epsilon, num_arms, context, true_weights, true_cost, budget)
#bandit_eg.run()
#print(bandit_eg.empirical_cost_means)
#print(bandit_eg.mu_hat)

# Plot comparison
#optimal_reward = bandit_eg.optimal_reward
#regret_eg = np.array(optimal_reward)

#plt.subplot(122)
#plt.plot(regret_eg.cumsum(), label='linear model')
#plt.title("Cumulative regret")
#plt.legend()
#plt.show()



# Kumulative Belohnung
