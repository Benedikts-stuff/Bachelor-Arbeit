import numpy as np
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from .models.gpr import MyGPR

class GPTS:
    def __init__(self, n_arms, context_dim, delta= 0.1):
        self.n_arms = n_arms
        self.n_features = context_dim

        self.kernels = [C(1.0, (1e-3, 1e3)) *RBF(length_scale=0.05, length_scale_bounds=(1e-5, 2)) for _ in range(n_arms)]
        self.gps = [
            MyGPR(kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=10)
            for kernel in self.kernels
        ]

        self.kernels_c = [C(1.0, (1e-3, 1e3)) * RBF(length_scale=0.05, length_scale_bounds=(1e-5, 2)) for _ in range(n_arms)]
        self.gps_c = [
            MyGPR(kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=10)
            for kernel in self.kernels_c
        ]

        self.arm_contexts = [[] for _ in range(n_arms)]
        self.arm_rewards = [[] for _ in range(n_arms)]
        self.arm_costs = [[] for _ in range(n_arms)]

        self.B = 0.2 #max kernel norm
        self.R = 1#max reward
        self.delta = delta # initialisierung
        self.gamma = 1e-8
        self.arm_counts = np.ones(n_arms)

    def compute_beta_t(self, gain):
        beta_t = self.B + self.R * np.sqrt(2*(gain + 1+ np.log(2/self.delta))) #gain is gamma
        return beta_t

    def select_arm(self, context, round):
        if round < self.n_arms:
            return round

        rewards = []
        costs = []
        for arm in range(self.n_arms):
            mu, sigma = self.gps[arm].predict(context.reshape(1, -1), return_std=True)
            mu_c, sigma_c = self.gps_c[arm].predict(context.reshape(1, -1), return_std=True)
            gain = 0.5 * np.log(1 + sigma ** 2 / 0.2**2)
            gain_c = 0.5 * np.log(1 + sigma_c ** 2 / 0.2**2)
            beta_t = self.compute_beta_t(gain)
            beta_t_c = self.compute_beta_t(gain_c)
            rewards.append(np.random.normal(np.clip(mu, 0, 1), beta_t * sigma))
            costs.append(np.random.normal(np.clip(mu_c, self.gamma, 1), beta_t_c * sigma_c))

        ratio = np.array(rewards)/np.array(costs)
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
