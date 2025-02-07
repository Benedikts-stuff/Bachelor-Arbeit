import numpy as np



class mLinUCB:
    def __init__(self, n_arms, context_dim, alpha):
        self.n_arms = n_arms
        self.n_features = context_dim
        self.gamma = 1e-8
        self.alpha = alpha
        self.arm_counts = np.ones(self.n_arms)
        self.A = np.array([np.identity(self.n_features) for _ in range(self.n_arms)])  # Covariance matrices for each arm
        self.b = np.zeros((self.n_arms, self.n_features))  # Linear predictors for each arm
        self.theta_hat = np.zeros((self.n_arms, self.n_features))  # Estimated theta for each arm (for debugging)

        #cost model
        self.b_c = np.zeros((self.n_arms, self.n_features))  # Linear predictors for each arm
        self.theta_hat_c = np.zeros((self.n_arms, self.n_features))  # Estimated theta for each arm


    def select_arm(self, context, round):
        expected_rewards = np.array([np.dot(self.theta_hat[a], context) for a in range(self.n_arms)])
        expected_cost = np.array([np.dot(self.theta_hat_c[a], context) for a in range(self.n_arms)])
        epsilon = np.array([self.alpha * np.sqrt(np.log(round+1)/ self.arm_counts[arm]) for arm in range(self.n_arms)])

        index = np.clip((expected_rewards + epsilon), None, 1)/np.clip(expected_cost+epsilon, self.gamma, None)

        return np.argmax(index)

    def update(self, reward, cost, chosen_arm, context):
        self.A[chosen_arm] += np.outer(context, context)
        self.b[chosen_arm] += reward * context
        self.b_c[chosen_arm] += cost * context
        self.theta_hat[chosen_arm] = np.linalg.inv(self.A[chosen_arm]).dot(self.b[chosen_arm])
        self.theta_hat_c[chosen_arm] = np.linalg.inv(self.A[chosen_arm]).dot(self.b_c[chosen_arm])

        self.arm_counts[chosen_arm] += 1

