import numpy as np
from matplotlib import pyplot as plt

class OmegaUCB_CDC:
    def __init__(self, n_actions, n_features, contexts, true_theta, cost, budget, repetition, logger, seed, p, cost_weight, reward_fn):
        """
        Initialize the OmegaUCB_CDC instance with parameters.
        logger should be None by default.
        n_actions: Number of arms (actions).
        n_features: Number of features for each context.
        contexts: Array of context vectors (data points).
        true_theta: True weight matrix (reward parameter) for each arm.
        cost: Cost per arm.
        budget: Total budget for playing arms.
        """
        np.random.seed(seed)
        self.n_actions = n_actions
        self.n_features = n_features
        self.contexts = contexts
        self.true_theta = true_theta
        self.cost = cost
        self.budget = budget
        self.og_budget = budget
        self.cum = np.zeros(self.n_actions)
        self.arm_counts = np.zeros(self.n_actions)

        self.empirical_cost_means = np.random.rand(self.n_actions)
        self.z = 1
        self.p = p
        self.repetition = repetition
        self.logger = logger
        self.summed_regret = 0
        self.cost_weight = cost_weight

        self.reward_fn = reward_fn
        self.cost_fn = None

        # Initialize variables
        self.A = np.array([np.identity(n_features) for _ in range(n_actions)])  # Covariance matrices for each arm
        self.b = np.zeros((n_actions, n_features))  # Linear predictors for each arm
        self.theta_hat = np.zeros((n_actions, n_features))  # Estimated theta for each arm
        self.choices = np.zeros(len(contexts), dtype=int)
        self.rewards = np.zeros(len(contexts))
        self.optimal_reward = []

        # Initialize cost model
        self.A_c = np.array([np.identity(n_features) for _ in range(n_actions)])  # Covariance matrices for each arm
        self.b_c = np.zeros((n_actions, n_features))  # Linear predictors for each arm
        self.theta_hat_c = np.zeros((n_actions, n_features))  # Estimated theta for each arm
        self.choices_c = np.zeros(len(contexts), dtype=int)
        self.costs_c = np.zeros(len(contexts))
        self.optimal_costs = []

    def set_cost_fn(self, func):
        self.cost_fn = func

    def calculate_upper_confidence_bound(self, context, round):
        """
        Calculate the upper confidence bound for a given action and context.
        """
        upper = []
        for i in range(self.n_actions):
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
            omega_r = (B / (2 * A)) + x
            upper.append(omega_r)

        return upper

    def calculate_lower_confidence_bound(self, context, round):
        """
        Calculate the lower confidence bound for a given action and context.
        """
        lower = []
        for i in range(self.n_actions):
            A_inv_c = np.linalg.inv(self.A_c[i])
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
            omega_r = (B / (2 * A)) - x
            lower.append(omega_r)

        return lower

    def select_arm(self, context, round):
        """
        Select the arm with the highest upper confidence bound, adjusted for cost.
        """
        upper = np.array(self.calculate_upper_confidence_bound(context, round))
        lower = np.array(self.calculate_lower_confidence_bound(context, round))
        ratio = upper / lower
        return np.argmax(ratio)

    def update_parameters(self, chosen_arm, context, actual_reward, actual_cost):
        """
        Update the parameters for the chosen arm based on observed context and reward.
        """
        self.A[chosen_arm] += np.outer(context, context)
        self.b[chosen_arm] += actual_reward * context
        self.theta_hat[chosen_arm] = np.linalg.inv(self.A[chosen_arm]).dot(self.b[chosen_arm])
        print("true theta: ", self.true_theta, " theta hat :", self.theta_hat)
        self.A_c[chosen_arm] += np.outer(context, context)
        self.b_c[chosen_arm] += actual_cost * context
        self.theta_hat_c[chosen_arm] = np.linalg.inv(self.A_c[chosen_arm]).dot(self.b_c[chosen_arm])

        self.cum[chosen_arm] += np.random.binomial(1, np.clip(actual_cost, 0, 1))
        self.empirical_cost_means[chosen_arm] = self.cum[chosen_arm] / (self.arm_counts[chosen_arm] + 1)
        self.budget -= self.cost[chosen_arm]

    def run(self):
        """
        Run the OmegaUCB algorithm over all rounds within the given budget.
        """
        i = 0
        while self.budget > np.max(self.cost):
            context = self.contexts[i]
            chosen_arm = self.select_arm(context, i)
            self.arm_counts[chosen_arm] += 1

            # Calculate reward and optimal reward using modular reward and cost functions
            actual_costs = np.array(self.cost_fn(context, self.cost_weight)) + np.random.normal(0, 0.0001, self.n_actions)
            actual_reward = np.array(self.reward_fn(context, self.true_theta)) + np.random.normal(0, 0.0001, self.n_actions)

            optimal_arm = np.argmax(actual_reward /actual_costs)

            self.rewards[i] = actual_reward[chosen_arm]/actual_costs[chosen_arm]
            self.optimal_reward.append(actual_reward[optimal_arm]/actual_costs[optimal_arm])

            # Update parameters for the chosen arm
            self.update_parameters(chosen_arm, context, actual_reward[chosen_arm], actual_costs[chosen_arm])
            self.choices[i] = chosen_arm

            self.summed_regret += (actual_reward[optimal_arm]/actual_costs[optimal_arm]) - (actual_reward[chosen_arm]/actual_costs[chosen_arm])

            if self.logger:
                self.logger.track_rep(self.repetition)
                self.logger.track_approach(0)
                self.logger.track_round(i)
                self.logger.track_regret(self.summed_regret)
                self.logger.track_normalized_budget((self.og_budget - self.budget) / self.og_budget)
                self.logger.track_spent_budget(self.og_budget - self.budget)
                self.logger.finalize_round()

            i += 1
        print('Finished OmegaUCB')

    def plot_results(self):
        """
        Plot the results showing the cumulative reward and convergence of norms.
        """
        plt.figure(figsize=(14, 6))

        # Plot cumulative reward
        plt.subplot(1, 2, 1)
        plt.plot(np.cumsum(self.optimal_reward) - np.cumsum(self.rewards[:len(self.optimal_reward)]), label='Cumulative regret')
        plt.xlabel('Rounds')
        plt.ylabel('Cumulative Reward')
        plt.legend()
        plt.title('Regret')

        plt.show()

