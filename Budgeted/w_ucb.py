import numpy as np


class OmegaUCB:
    def __init__(self, K, B, cost, context, theta, rho=0.25, eta_r=None, eta_c=None):
        self.K = K  # Number of arms
        self.n_features = 3
        self.B = B  # Total budget
        self.rho = rho  # Confidence interval scaling parameter
        self.cost = cost
        self.context = context

        # Reward and cost variance parameters for each arm, default to 1 if not provided
        self.eta_r = eta_r if eta_r is not None else np.ones(K)
        self.eta_c = eta_c if eta_c is not None else np.ones(K)

        # Initialize counts and estimates for each arm
        self.counts = np.zeros(K)  # Track the number of times each arm is played
        self.mu_r = np.zeros(K)  # Estimated reward mean for each arm
        self.mu_c = np.zeros(K)  # Estimated cost mean for each arm
        self.t = 0  # Time step
        self.z = np.sqrt(2*self.rho*np.log(self.t))
        self.arm_played = np.zeros(K)

        self.M = 1
        self.m = 0

        self.A = np.array([np.identity(self.K) for _ in range(self.K)])  # Covariance matrices for each arm
        self.b = np.zeros((self.K, self.n_features))  # Linear predictors for each arm
        self.theta_hat = np.zeros((self.K, self.n_features))  # Estimated theta for each arm
        self.theta = theta
        self.cumulative_cost = np.zeros(self.K)
        self.cost_history = []
        self.reward_history = []

        self.t = 0
        self.adaptive = False

    def compute_eta(self, mean, variance, n: int, m=0, M=1, min_samples=30) -> float:
        if n < min_samples:
            return 1.0
        bernoulli_variance = (M - mean) * (mean - m)
        bernoulli_sample_variance = n / (n - 1) * bernoulli_variance
        if bernoulli_variance == 0:
            return 1.0
        eta = variance / bernoulli_sample_variance
        return min(eta, 1.0)

    def _adaptive_z(self):
        if self.t == 0:
            return 0
        else:
            z = np.sqrt(2 * self.rho * np.log(self.t + 1))
        return z


    def compute_omega_reward(self,arm, context):
        A_inv = np.linalg.inv(self.A[arm])  # A^-1
        theta_hat = A_inv.dot(self.b[arm])  # theta_hat = A^-1 * b
        mu_r = theta_hat[arm].dot(context) # theta_hat * context = empirical reward
        eta_r = self.compute_eta(mu_r, context.dot(A_inv).dot(context), self.arm_played[arm])

        if self.adaptive:
            z = self._adaptive_z()
        else:
            z = self.z

        A = self.arm_played[arm] + z**2 * eta_r
        B = (2 * self.arm_played[arm] * mu_r) + z**2 * eta_r*(self.M + self.m)
        C = self.arm_played[arm] * mu_r**2 + self.z**2 * eta_r*(self.M * self.m)
        omega_r = (B/(2*A)) + np.sqrt(B**2/(4*A**2) - C/A)
        return omega_r

    def compute_omega_cost(self,arm, context):
        n = self.arm_played[arm]
        mu_c = self.cumulative_cost[arm] / n
        eta_c = self.compute_eta(mu_c, np.var(self.cost_history, ddof=1), n=n, min_samples=30)

        if self.adaptive:
            z = self._adaptive_z()
        else:
            z = self.z

        A = self.arm_played[arm] + z**2 * eta_c
        B = (2 * self.arm_played[arm] * mu_c) + z**2 * eta_c*(self.M + self.m)
        C = self.arm_played[arm] * mu_c**2 + z**2 * eta_c*(self.M * self.m)
        omega_c = (B/(2*A)) - np.sqrt(B**2/(4*A**2) - C/A)
        return omega_c


    def run(self):
        i = 0
        while self.B > np.max(self.cost):
            x_i = self.context[i]
            ratios = []
            for arm in range(self.K):
                upper_cb_rw =self.compute_omega_reward(arm, x_i)
                lower_cb_cost = self.compute_omega_cost(arm, x_i)
                ratios.append(upper_cb_rw/lower_cb_cost)

            arm = np.argmax(ratios)
            true_reward =self.theta[arm].dot(x_i)
            self.reward_history.append(true_reward)


            i += 1

