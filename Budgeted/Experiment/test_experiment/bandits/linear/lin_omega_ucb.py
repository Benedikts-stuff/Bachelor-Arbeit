import numpy as np

class LinOmegaUCB:
    def __init__(self, n_arms, context_dim, p):
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.arm_counts = np.zeros(self.n_arms)
        self.z = 1
        self.p = p
        self.gamma =1e-8
        self.A = np.array([np.identity(self.context_dim) for _ in range(self.n_arms)])
        self.b = np.zeros((self.n_arms,self.context_dim))
        self.theta_hat = np.zeros((self.n_arms, self.context_dim))
        self.b_c = np.zeros((self.n_arms,self.context_dim))
        self.theta_hat_c = np.zeros((self.n_arms,self.context_dim))


    def calculate_upper_confidence_bound(self, context, round):
        upper = []
        for i in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[i])
            theta_hat = A_inv.dot(self.b[i])
            variance = context.dot(A_inv).dot(context)
            mu_r = theta_hat.dot(context)

            eta = 1
            arm_count = self.arm_counts[i]
            z = np.sqrt(2 * self.p * np.log(round + 2))

            A = arm_count + z**2 * eta
            B = 2 * arm_count * mu_r + z**2 * eta
            C = arm_count * mu_r**2
            x = np.sqrt((B**2 / (4 * A**2)) - (C / A))
            omega_r = np.clip((B / (2 * A)) + x, 0, 1)
            upper.append(omega_r)


        return upper

    def calculate_lower_confidence_bound(self, context, round):
        """
        Calculate the lower confidence bound for a given action and context.
        """
        lower = []
        for i in range(self.n_arms):
            A_inv_c = np.linalg.inv(self.A[i])
            theta_hat_c = A_inv_c.dot(self.b_c[i])
            variance = context.dot(A_inv_c).dot(context)
            mu_r = theta_hat_c.dot(context)

            eta = 1
            arm_count = self.arm_counts[i]
            z = np.sqrt(2 * self.p * np.log(round + 2))

            A = arm_count + z**2 * eta
            B = 2 * arm_count * mu_r + z**2 * eta
            C = arm_count * mu_r**2
            x = np.sqrt((B**2 / (4 * A**2)) - (C / A))
            omega_r = np.clip((B / (2 * A)) - x, self.gamma, 1)
            lower.append(omega_r)

        return lower

    def select_arm(self, context, round):
        if round < self.n_arms:
            return round # play each arm once

        upper = np.array(self.calculate_upper_confidence_bound(context, round))
        lower = np.array(self.calculate_lower_confidence_bound(context, round))
        ratio = upper / lower
        return np.argmax(ratio)

    def update(self,actual_reward, actual_cost, chosen_arm, context):
        self.A[chosen_arm] += np.outer(context, context)
        self.b[chosen_arm] += actual_reward * context
        self.theta_hat[chosen_arm] = np.linalg.inv(self.A[chosen_arm]).dot(self.b[chosen_arm])
        self.b_c[chosen_arm] += actual_cost * context
        self.theta_hat_c[chosen_arm] = np.linalg.inv(self.A[chosen_arm]).dot(self.b_c[chosen_arm])
        self.arm_counts[chosen_arm] += 1



