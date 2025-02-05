import numpy as np


class ThompsonSampling:
    def __init__(self, n_arms, context_dim, v):
        self.n_features = context_dim
        self.variance = v
        self.n_arms = n_arms

        self.gamma = 1e-8

        # Initialisiere die Parameter für jeden Arm
        self.B = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])
        self.f = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        self.f_c = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        self.mu_hat = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        self.mu_hat_c = [np.zeros(self.n_features) for _ in range(self.n_arms)]


    def sample_mu(self, mean):
        return np.array([
            np.random.multivariate_normal(mean[arm], self.variance**2 * np.linalg.inv(self.B[arm]))
            for arm in range(self.n_arms)
        ])

    def select_arm(self, context, round):
        theta_hat_r = self.sample_mu(self.mu_hat)
        theta_hat_c = self.sample_mu(self.mu_hat_c)

        rewards = np.clip(np.array([np.dot(theta_hat_r[arm], context) for arm in range(self.n_arms)]), 0, None)
        cost = np.clip(np.array([np.dot(theta_hat_c[arm], context) for arm in range(self.n_arms)]), self.gamma, None)

        return np.argmax(rewards /cost)


    def update(self, reward, cost, chosen_arm, context):
        self.B[chosen_arm] += np.outer(context, context)
        self.f[chosen_arm] += reward * context
        self.f_c[chosen_arm] += cost * context
        self.mu_hat[chosen_arm] = np.linalg.inv(self.B[chosen_arm]).dot(self.f[chosen_arm])
        self.mu_hat_c[chosen_arm] = np.linalg.inv(self.B[chosen_arm]).dot(self.f_c[chosen_arm])


