import numpy as np
from sklearn.gaussian_process.kernels import RBF
from .models.gpr import MyGPR

class Beta_GPTS:
    def __init__(self, n_arms, context_dim, delta= 0.1):
        self.n_arms = n_arms
        self.n_features = context_dim

        self.kernels = [RBF(length_scale=0.2, length_scale_bounds=(1e-60, 10)) for _ in range(n_arms)]
        self.gps = [
            MyGPR(kernel=kernel, alpha=1e-2, normalize_y=True, n_restarts_optimizer=10)
            for kernel in self.kernels
        ]

        self.kernels_c = [RBF(length_scale=0.2, length_scale_bounds=(1e-60, 10)) for _ in range(n_arms)]
        self.gps_c = [
            MyGPR(kernel=kernel, alpha=1e-2, normalize_y=True, n_restarts_optimizer=10)
            for kernel in self.kernels_c
        ]

        self.arm_contexts = [[] for _ in range(n_arms)]
        self.arm_rewards = [[] for _ in range(n_arms)]
        self.arm_costs = [[] for _ in range(n_arms)]

        self.B = 0.2 #max kernel norm
        self.R = 1#max reward
        self.delta = delta # initialisierung
        self.gamma = 1e-8
        self.noise_var = 0.01

        self.sigma_t_1 = np.array([2.5 for _ in range(self.n_arms)])
        self.sigma_t_1_c = np.array([2.5 for _ in range(self.n_arms)])

        self.arm_counts = np.ones(n_arms)

    def select_arm(self, context, round):
        if round < self.n_arms:
            return round

        rewards = []
        costs = []
        for arm in range(self.n_arms):
            mu, sigma = self.gps[arm].predict(context.reshape(1, -1), return_std=True)
            mu_c, sigma_c = self.gps_c[arm].predict(context.reshape(1, -1), return_std=True)

            gain = 0.5 * np.log(1 + sigma ** 2 / self.noise_var**2)
            gain_c = 0.5 * np.log(1 + sigma_c ** 2 / self.noise_var**2)

            # Kombinieren von Mean und Bonus (hier als Beispiel Beta-Verteilung, ähnlich wie dein Ansatz)
            sampled_reward = np.clip(np.random.beta(np.exp(1/gain) *self.arm_counts[arm] * mu,
                                            np.exp(1/gain) * self.arm_counts[arm] * (1-mu)), self.gamma, 1-self.gamma)
            sampled_cost = np.clip(np.random.beta(np.exp(1/gain_c) * self.arm_counts[arm] * mu_c,
                                            np.exp(1/gain_c) * self.arm_counts[arm] * (1-mu_c)), self.gamma, 1-self.gamma)

            rewards.append(sampled_reward)
            costs.append(sampled_cost)

        ratio = np.array(rewards) / np.array(costs)
        return np.argmax(ratio)

    def update(self, reward, cost, chosen_arm,  context):
        self.arm_contexts[chosen_arm].append(context)
        self.arm_rewards[chosen_arm].append(reward)
        self.arm_costs[chosen_arm].append(cost)
        self.arm_counts[chosen_arm] += 1

        if sum(len(v) for v in self.arm_contexts) < 500:
            self.gps[chosen_arm].fit(
                np.array(self.arm_contexts[chosen_arm]),
                np.array(self.arm_rewards[chosen_arm])
            )

            self.gps_c[chosen_arm].fit(
                np.array(self.arm_contexts[chosen_arm]),
                np.array(self.arm_costs[chosen_arm])
            )
