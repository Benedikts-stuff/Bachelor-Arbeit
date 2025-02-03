import numpy as np


class BetaThompsonSampling:
    def __init__(self, n_arms, context_dim):
        self.n_features = context_dim
        self.n_arms = n_arms

        self.arm_counts = np.ones(self.n_arms)
        self.gamma = 1e-8

        # Initialisiere die Parameter für jeden Arm
        self.B = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])
        self.f = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        self.f_c = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        self.mu_hat = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        self.mu_hat_c = [np.zeros(self.n_features) for _ in range(self.n_arms)]


    def select_arm(self, context, round):
        if round < self.n_arms:
            return round

        # anstatt self.arm_counts[arm] nehmen für bessere perfromance,
        # für bernoulli bandits rewards self.arm_counts[arm] mal bernoulli samplen und summieren
        rewards = np.clip(np.array([np.dot(self.mu_hat[arm], context) for arm in range(self.n_arms)]), self.gamma,1 - self.gamma)
        sampled_reward = np.array([np.random.beta(self.arm_counts[arm] * (rewards[arm]), self.arm_counts[arm] * (1 - rewards[arm]), 1) for arm in range(self.n_arms)])
        cost = np.clip(np.array([np.dot(self.mu_hat_c[arm], context) for arm in range(self.n_arms)]), self.gamma,1 - self.gamma)
        sampled_cost = np.array([np.random.beta(self.arm_counts[arm] * cost[arm], self.arm_counts[arm] * (1 - cost[arm]), 1) for arm in range(self.n_arms)])

        return np.argmax(sampled_reward /(sampled_cost + self.gamma))


    def update(self, reward, cost, chosen_arm, context):
        self.B[chosen_arm] += np.outer(context, context)
        self.f[chosen_arm] += reward * context
        self.f_c[chosen_arm] += cost * context
        self.mu_hat[chosen_arm] = np.linalg.inv(self.B[chosen_arm]).dot(self.f[chosen_arm])
        self.mu_hat_c[chosen_arm] = np.linalg.inv(self.B[chosen_arm]).dot(self.f_c[chosen_arm])
        self.arm_counts[chosen_arm] +=1



