import numpy as np
from sklearn.gaussian_process.kernels import RBF
from .models.gpr import MyGPR
from sklearn.gaussian_process.kernels import ConstantKernel as C

class GPWUCB:
    def __init__(self, n_arms, context_dim, p):
        self.n_arms = n_arms
        self.n_features = context_dim
        self.beta_t = 1
        self.gamma= 1e-8

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

        self.arm_counts = np.ones(self.n_arms)

        self.p =p


    def calculate_upper_confidence_bound(self, context, round):
        """
        Calculate the upper confidence bound for a given action and context.
        """
        upper =[]
        for i in range(self.n_arms):
            mu_r,sigma= self.gps[i].predict(context.reshape(1, -1), return_std=True)
            #print(f"NeuralOmnegaUCB mu_r in round {round} and arm {i}", mu_r)
            eta = 1
            arm_count = self.arm_counts[i]
            z = np.sqrt(2* self.p* np.log(round + 2))
            if mu_r != 0 and mu_r != 1:
                eta = 1

            A = arm_count + z**2 * eta
            B = 2*arm_count*mu_r + z**2 * eta # eig noch * (M-m) aber das ist hier gleich (1-0)
            C = arm_count* mu_r**2
            gain = 0.5 * np.log(1 + sigma ** 2 / 0.01 ** 2)
            x = np.sqrt(np.clip((B**2 / (4* A**2)) - (C/A), 0, None))
            ucb = np.clip((B/(2*A)) + gain * x, 0, 1)
            upper.append(ucb[0])

        # Adjust for cost and return estimated reward per cost ratio
        return np.array(upper)

    def calculate_lower_confidence_bound(self,context, round):
        lower =[]
        for i in range(self.n_arms):
            mu_c,sigma= self.gps_c[i].predict(context.reshape(1, -1), return_std=True)
            eta = 1
            arm_count = self.arm_counts[i]
            z = np.sqrt(2* self.p* np.log(round + 2))
            if mu_c != 0 and mu_c != 1:
                eta = 1

            A = arm_count + z**2 * eta
            B = 2*arm_count*mu_c + z**2 * eta # eig. * (M-m) aber das ist hier gleich (1-0)
            C = arm_count* mu_c**2
            x = np.sqrt(np.clip((B**2 / (4* A**2)) - (C/A), 0, None))
            gain = 0.5 * np.log(1 + sigma ** 2 / 0.01** 2)

            omega_r = np.clip((B/(2*A)) - gain * x, self.gamma, 1)

            lower.append(omega_r[0])

        return np.array(lower)

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
        self.arm_costs[chosen_arm].append(cost)
        self.arm_counts[chosen_arm] +=1

        if sum(len(v) for v in self.arm_contexts) < 500:
            self.gps[chosen_arm].fit(
                np.array(self.arm_contexts[chosen_arm]),
                np.array(self.arm_rewards[chosen_arm])
            )

            self.gps_c[chosen_arm].fit(
                np.array(self.arm_contexts[chosen_arm]),
                np.array(self.arm_costs[chosen_arm])
            )
