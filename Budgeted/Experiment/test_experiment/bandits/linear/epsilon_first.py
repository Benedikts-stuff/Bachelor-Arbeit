import numpy as np

class LinFirst:
    def __init__(self, context_dim, n_arms, epsilon, budget):
        self.n_features = context_dim
        self.alpha = epsilon
        self.n_arms = n_arms
        self.total_budget = budget
        self.exploration_budget = self.alpha * self.total_budget
        self.B = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])
        self.f = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        self.f_c = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        self.theta_hat = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        self.theta_hat_c = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        self.arm_counts = np.zeros(self.n_arms)
        self.gamma = 1e-8

    def select_arm(self, context, i):
        if self.exploration_budget > 0:
            return np.random.choice(self.n_arms)
        else:
            expected_rewards = np.array([np.clip(np.dot(self.theta_hat[a], context), self.gamma, None) for a in range(self.n_arms)])
            expected_cost = np.array([np.clip(np.dot(self.theta_hat_c[a], context), self.gamma, None) for a in range(self.n_arms)])

            return np.argmax(expected_rewards / expected_cost)

    def update(self, reward, cost, chosen_arm, context):
        self.B[chosen_arm] += np.outer(context, context)
        self.f[chosen_arm] += reward * context
        self.f_c[chosen_arm] += cost * context
        self.theta_hat[chosen_arm] = np.linalg.inv(self.B[chosen_arm]).dot(self.f[chosen_arm])
        self.theta_hat_c[chosen_arm] = np.linalg.inv(self.B[chosen_arm]).dot(self.f_c[chosen_arm])
        self.exploration_budget -= cost
        self.arm_counts[chosen_arm] += 1
