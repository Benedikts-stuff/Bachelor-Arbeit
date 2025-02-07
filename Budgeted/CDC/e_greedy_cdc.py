import numpy as np
import matplotlib.pyplot as plt


class EGreedy_CDC:
    def __init__(self, d, epsilon,  n_arms, contexts, true_weights, true_cost, budget, logger, repetition, seed, cost_weights, cost_fn, reward_fn):
        """
        d: Dimension der Kontextvektoren
        epsilon: Wahrscheinlichkeit für Exploration
        """
        np.random.seed(seed)
        self.repetition = repetition
        self.logger = logger
        self.n_features = d
        self.epsilon = epsilon
        self.n_arms = n_arms
        #reward model
        self.B = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])
        self.f = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        self.mu_hat = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        #cost model
        self.B_c = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])
        self.f_c = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        self.mu_hat_c = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        self.cost_weights = cost_weights
        self.reward_fn = reward_fn
        self.cost_fn = cost_fn

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
        self.optimal_reward = []
        self.summed_regret = 0

    def select_arm(self, context):
        epsilon = min(1, self.epsilon * (self.n_arms/(self.i +1)))
        print(epsilon)
        if np.random.rand() < epsilon:
            return np.random.choice(self.n_arms)  # Exploration
        else:
            expected_rewards = np.array([np.dot(self.mu_hat[i], context) for i in range(self.n_arms)])
            expected_cost = np.array([np.dot(self.mu_hat_c[i], context) for i in range(self.n_arms)])
            return np.argmax(expected_rewards / (expected_cost + self.gamma))  # Exploitation

    def update(self, reward, chosen_arm, context, cost):
        self.B[chosen_arm] += np.outer(context, context)
        self.f[chosen_arm] += reward * context
        self.mu_hat[chosen_arm] = np.linalg.inv(self.B[chosen_arm]).dot(self.f[chosen_arm])

        self.B_c[chosen_arm] += np.outer(context, context)
        self.f_c[chosen_arm] += cost * context
        self.mu_hat_c[chosen_arm] = np.linalg.inv(self.B_c[chosen_arm]).dot(self.f_c[chosen_arm])
        print("true theta c: ", self.cost_weights, " theta hat c :", self.mu_hat_c)

        self.cum[chosen_arm] += np.random.binomial(1, self.costs[chosen_arm])
        self.empirical_cost_means[chosen_arm] =  (self.cum[chosen_arm])/ (self.arm_counts[chosen_arm] +1)

        self.arm_counts[chosen_arm] += 1

        self.budget = self.budget - self.costs[chosen_arm]

    def run(self):
        self.i = 0
        count = 0
        while self.budget > self.max_cost:
            context = self.contexts[self.i]
            chosen_arm = self.select_arm(context)
            reward = np.array(self.reward_fn(context, self.true_weights, self.i))
            costs = np.array(self.cost_fn(context, self.cost_weights, self.i))

            self.update(reward[chosen_arm], chosen_arm, context, costs[chosen_arm])
            observed_reward = reward[chosen_arm] / costs[chosen_arm]
            self.observed_reward_history.append(observed_reward)

            optimal = np.max(reward / costs)
            opt_arm = np.argmax(reward / costs)
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




