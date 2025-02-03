import numpy as np
from sklearn.gaussian_process.kernels import RBF
from .models.gpr import MyGPR
from sklearn.gaussian_process.kernels import ConstantKernel as C

class GPUCB:
    def __init__(self, n_arms, context_dim, gamma):
        self.n_arms = n_arms
        self.gamma = gamma
        self.delta = 1e-8
        self.beta_t = 1

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
        self.arm_cost = [[] for _ in range(n_arms)]

        self.arm_counts = np.ones(self.n_arms)
        self.B = 0.2 #max kernel norm
        self.R = 1#function range



    def calculate_upper_confidence_bound(self, context, round):
        ucb_values = []
        for arm_id in range(self.n_arms):
            mu, sigma = self.gps[arm_id].predict(context.reshape(1, -1), return_std=True)
            beta = 2 * np.log(self.arm_counts[arm_id] * ((round+1)**2) * np.pi ** 2 / (6 * self.gamma))
            ucb = np.clip(mu + (np.sqrt(beta / 5) * sigma), 0, 1)
            ucb_values.append(ucb[0])

        return np.array(ucb_values)

    def calculate_lower_confidence_bound(self, context, round):
        lcb_values = []
        for arm_id in range(self.n_arms):
            mu, sigma = self.gps_c[arm_id].predict(context.reshape(1, -1), return_std=True)
            beta = 2 * np.log(self.arm_counts[arm_id] * ((round+1) ** 2) * np.pi ** 2 / (6 * self.gamma))
            lcb = np.clip(mu - (np.sqrt(beta / 5) * sigma), self.delta, 1)
            lcb_values.append(lcb[0])

        return np.array(lcb_values)


    def select_arm(self, context, round):
        if round < self.n_arms:
            return round
        upper =self.calculate_upper_confidence_bound(context, round)
        lower =self.calculate_lower_confidence_bound(context, round)
        arm = np.argmax(upper/lower)
        return arm

    def update(self, reward, cost, chosen_arm, context):
        self.arm_contexts[chosen_arm].append(context)
        self.arm_rewards[chosen_arm].append(reward)
        self.arm_cost[chosen_arm].append(cost)
        self.arm_counts[chosen_arm] +=1

        if sum(len(v) for v in self.arm_contexts) < 300:
            self.gps[chosen_arm].fit(
                np.array(self.arm_contexts[chosen_arm]),
                np.array(self.arm_rewards[chosen_arm])
            )

            self.gps_c[chosen_arm].fit(
                np.array(self.arm_contexts[chosen_arm]),
                np.array(self.arm_cost[chosen_arm])
            )