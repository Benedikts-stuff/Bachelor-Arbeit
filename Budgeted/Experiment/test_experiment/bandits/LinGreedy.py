import numpy as np





class LinGreedy:
    def __init__(self, context_dim, n_arms, epsilon, seed):
        np.random.seed(seed)
        self.n_features = context_dim
        self.epsilon = epsilon
        self.n_arms = n_arms
        self.B = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])
        self.f = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        self.mu_hat = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        self.empirical_cost_means = np.random.rand(self.n_arms)
        self.arm_counts = np.zeros(self.n_arms)
        self.gamma = 0.00000001
        self.cum = np.zeros(self.n_arms)


    def select_action(self, context, i):
        epsilon = min(1, self.epsilon * (self.n_arms/(i +1))) # für bernoulli epsilon = 5 nehmen
        if np.random.rand() < epsilon:
            return np.random.choice(self.n_arms)  # Exploration
        else:
            expected_rewards = np.array([np.dot(self.mu_hat[i], context) for i in range(self.n_arms)])
            return np.argmax(expected_rewards / (self.empirical_cost_means + self.gamma))  # Exploitation

    def update(self, reward, cost, chosen_arm, context):
        self.B[chosen_arm] += np.outer(context, context)
        self.f[chosen_arm] += reward * context
        self.mu_hat[chosen_arm] = np.linalg.inv(self.B[chosen_arm]).dot(self.f[chosen_arm])

        self.cum[chosen_arm] += cost
        self.empirical_cost_means[chosen_arm] =  (self.cum[chosen_arm])/ (self.arm_counts[chosen_arm] +1)

        self.arm_counts[chosen_arm] += 1




