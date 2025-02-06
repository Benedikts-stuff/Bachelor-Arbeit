import numpy as np
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from .non_linear.models.gpr import MyGPR


class EXP4:
    def __init__(self, context_dim, n_arms, eta=0.1, epsilon=0.1):

        self.n_features = context_dim
        self.n_arms = n_arms
        self.n_experts = 2
        self.eta = eta
        self.epsilon = epsilon
        self.gamma = 1e-8
        self.exp_played = 0

        self.weights = np.ones(self.n_experts)

        self.B = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])
        self.f = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        self.f_c = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        self.theta_hat = [np.zeros(self.n_features) for _ in range(self.n_arms)]
        self.theta_hat_c = [np.zeros(self.n_features) for _ in range(self.n_arms)]

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

        self.last_expert_advice = None  # shape: (n_experts, n_arms)
        self.last_combined_prob = None  # shape: (n_arms,)

    def softmax(self, x):
        ex = np.exp(x - np.max(x))
        return ex / np.sum(ex)

    def compute_utilities(self, context):

        utilities_linear = np.zeros(self.n_arms)
        utilities_gp = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            reward_linear = np.dot(context, self.theta_hat[a])
            cost_linear = np.dot(context, self.theta_hat_c[a])
            utilities_linear[a] = reward_linear/(cost_linear+self.gamma)

            reward_gp = self.gps[a].predict(context.reshape(1, -1))
            print("guesses: ", reward_gp)
            cost_gp = self.gps_c[a].predict(context.reshape(1, -1))
            reward_gp = reward_gp.item() if reward_gp.size == 1 else reward_gp
            cost_gp = cost_gp.item() if cost_gp.size == 1 else cost_gp
            utilities_gp[a] = reward_gp/(cost_gp + self.gamma)

        return utilities_linear, utilities_gp

    def select_arm(self, context, round):

        util_linear, util_gp = self.compute_utilities(context)

        p_linear = self.softmax(util_linear)
        p_gp = self.softmax(util_gp)

        expert_advices = np.vstack([p_linear, p_gp])
        self.last_expert_advice = expert_advices

        total_weight = np.sum(self.weights)
        weighted_advice = np.dot(self.weights / total_weight, expert_advices)  # shape: (n_arms,)

        combined_prob = (1 - self.epsilon) * weighted_advice + self.epsilon / self.n_arms
        self.last_combined_prob = combined_prob

        print("probs: ", combined_prob)
        chosen_arm = np.random.choice(self.n_arms, p=combined_prob)
        return chosen_arm

    def update(self, reward, cost, chosen_arm, context):

        if self.last_expert_advice is None or self.last_combined_prob is None:
            raise ValueError("select_arm muss vor update aufgerufen werden.")
        p_a = self.last_combined_prob[chosen_arm]
        for i in range(self.n_experts):
            p_i = self.last_expert_advice[i, chosen_arm]
            estimated_reward = reward * (p_i / (p_a + self.gamma))
            self.weights[i] *= np.exp(self.eta * estimated_reward)

        self.B[chosen_arm] += np.outer(context, context)
        self.f[chosen_arm] += reward * context
        self.f_c[chosen_arm] += cost * context

        self.theta_hat[chosen_arm] = np.linalg.inv(self.B[chosen_arm]).dot(self.f[chosen_arm])
        self.theta_hat_c[chosen_arm] = np.linalg.inv(self.B[chosen_arm]).dot(self.f_c[chosen_arm])

        self.arm_contexts[chosen_arm].append(context)
        self.arm_rewards[chosen_arm].append(reward)
        self.arm_costs[chosen_arm].append(cost)

        total_points = sum(len(v) for v in self.arm_contexts)
        if total_points < 500:
            self.gps[chosen_arm].fit(
                np.array(self.arm_contexts[chosen_arm]),
                np.array(self.arm_rewards[chosen_arm])
            )

            self.gps_c[chosen_arm].fit(
                np.array(self.arm_contexts[chosen_arm]),
                np.array(self.arm_costs[chosen_arm])
            )

        self.last_expert_advice = None
        self.last_combined_prob = None


