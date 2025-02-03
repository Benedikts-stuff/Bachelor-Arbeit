import numpy as np



class LinCUCB:
    def __init__(self, n_arms, context_dim):
        self.n_arms = n_arms
        self.n_features = context_dim
        self.gamma = 1e-8
        self.arm_counts = np.zeros(self.n_arms)
        self.A = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])  # Covariance matrices for each arm
        self.b = np.zeros((self.n_arms, self.n_features))  # Linear predictors for each arm
        self.theta_hat = np.zeros((self.n_arms, self.n_features))  # Estimated theta for each arm (for debugging)

        #cost model
        self.b_c = np.zeros((self.n_arms, self.n_features))  # Linear predictors for each arm
        self.theta_hat_c = np.zeros((self.n_arms, self.n_features))  # Estimated theta for each arm


    def select_arm(self, context, round):
        ratio = []
        for arm in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[arm])
            expected_rewards = np.dot(self.theta_hat[arm], context)
            expected_cost = np.dot(self.theta_hat_c[arm], context)
            uncertainty = context.dot(A_inv).dot(context)
            ci = (1 + np.sqrt(np.log(2 * (round + 1)) / 2)) * np.sqrt(uncertainty)

            upper = expected_rewards + ci
            lower = np.clip(expected_cost, self.gamma, None)

            ratio.append(upper / lower)
        return np.argmax(ratio)

    def update(self, reward, cost, chosen_arm, context):
        self.A[chosen_arm] += np.outer(context, context)
        self.b[chosen_arm] += reward * context
        self.b_c[chosen_arm] += cost * context
        self.theta_hat[chosen_arm] = np.linalg.inv(self.A[chosen_arm]).dot(self.b[chosen_arm])
        self.theta_hat_c[chosen_arm] = np.linalg.inv(self.A[chosen_arm]).dot(self.b_c[chosen_arm])

        self.arm_counts[chosen_arm] += 1

