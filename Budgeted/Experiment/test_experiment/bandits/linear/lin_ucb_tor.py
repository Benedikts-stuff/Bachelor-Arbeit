import numpy as np



class TorLinUCB:
    def __init__(self, n_arms, context_dim, budget):
        self.n_arms = n_arms
        self.n_features = context_dim
        self.gamma = 1e-8
        self.arm_counts = np.ones(self.n_arms)
        self.A = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])  # Covariance matrices for each arm
        self.b = np.zeros((self.n_arms, self.n_features))  # Linear predictors for each arm
        self.theta_hat = np.zeros((self.n_arms, self.n_features))  # Estimated theta for each arm (for debugging)

        #cost model
        self.b_c = np.zeros((self.n_arms, self.n_features))  # Linear predictors for each arm
        self.theta_hat_c = np.zeros((self.n_arms, self.n_features))  # Estimated theta for each arm


    def select_arm(self, context, round):
        A_inv = np.array([np.linalg.inv(self.A[a]) for a in range(self.n_arms)])
        expected_rewards = np.array([np.dot(self.theta_hat[a], context) for a in range(self.n_arms)])
        expected_cost = np.array([np.dot(self.theta_hat_c[a], context) for a in range(self.n_arms)])
        a= (round + 1)*np.log(round+1) +1
        alpha = np.sqrt(np.clip(2 * np.log(a) / self.arm_counts, 0, None))

        ci = alpha

        upper = np.clip(expected_rewards + ci, 0, None)
        lower = expected_cost -ci

        ratio = upper / (lower+ self.gamma)
        return np.argmax(ratio)

    def update(self, reward, cost, chosen_arm, context):
        self.A[chosen_arm] += np.outer(context, context)
        self.b[chosen_arm] += reward * context
        self.b_c[chosen_arm] += cost * context
        self.theta_hat[chosen_arm] = np.linalg.inv(self.A[chosen_arm]).dot(self.b[chosen_arm])
        self.theta_hat_c[chosen_arm] = np.linalg.inv(self.A[chosen_arm]).dot(self.b_c[chosen_arm])

        self.arm_counts[chosen_arm] += 1

