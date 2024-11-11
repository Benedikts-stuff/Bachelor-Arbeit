import numpy as np
from matplotlib import pyplot as plt



class LinUCB:
    def __init__(self, n_actions, n_features, contexts, true_theta, cost, alpha, budget, logger, repetion, seed):
        """
        Initialize the LinUCB instance with parameters.

        n_actions: Number of arms (actions).
        n_features: Number of features for each context.
        contexts: Array of context vectors (data points).
        true_theta: True weight matrix (reward parameter) for each arm.
        cost: Cost per arm.
        alpha: Exploration parameter for the upper confidence bound.
        budget: Total budget for playing arms.
        """
        np.random.seed(seed)
        self.logger = logger
        self.repetition = repetion
        self.n_actions = n_actions
        self.n_features = n_features
        self.contexts = contexts - 0.5
        self.true_theta = true_theta
        self.cost = cost
        self.alpha = alpha
        self.budget = budget
        self.og_budget = budget
        self.cum = np.zeros(self.n_actions)
        self.arm_counts = np.zeros(self.n_actions)
        self.gamma = 0.00000001

        self.empirical_cost_means = np.random.rand(self.n_actions)
        self.summed_regret = 0

        # Initialize variables
        self.A = np.array([np.identity(n_features) for _ in range(n_actions)])  # Covariance matrices for each arm
        self.b = np.zeros((n_actions, n_features))  # Linear predictors for each arm
        self.theta_hat = np.zeros((n_actions, n_features))  # Estimated theta for each arm
        self.choices = np.zeros(len(contexts), dtype=int)
        self.rewards = np.zeros(len(contexts))
        self.optimal_reward = []
        self.norms = np.zeros(len(contexts))

    def calculate_upper_confidence_bound(self, action, context, count):
        """
        Calculate the upper confidence bound for a given action and context.
        """
        A_inv = np.linalg.inv(self.A[action])
        theta_hat = A_inv.dot(self.b[action])
        mean_estimate = theta_hat.dot(context)
        ta = context.dot(A_inv).dot(context)
        upper_ci =(1 + np.sqrt(np.log(2 * (count + 1)) / 2)) * np.sqrt(ta)

        # Adjust for cost and return estimated reward per cost ratio
        return (mean_estimate + upper_ci)

    def calculate_lower_confidence_bound(self, action, count):
        """
        Calculate the upper confidence bound for a given action and context.
        """
        mean = self.empirical_cost_means[action]
        lower_cb =self.gamma * np.sqrt(np.log(count + 1)/ (self.arm_counts[action] + 1))

        # Adjust for cost and return estimated reward per cost ratio
        return (mean + lower_cb)

    def select_arm(self, context, count):
        """
        Select the arm with the highest upper confidence bound, adjusted for cost.
        """
        p = np.array([self.calculate_upper_confidence_bound(a, context, count) for a in range(self.n_actions)])
        l = np.array([self.calculate_lower_confidence_bound(a, count) for a in range(self.n_actions)])
        p += np.random.random(len(p)) * 0.000001  # Avoid bias with tie-breaking by adding small noise
        res = p/l
        return np.argmax(res)

    def update_parameters(self, chosen_arm, context, actual_reward):
        """
        Update the parameters for the chosen arm based on observed context and reward.
        """
        self.A[chosen_arm] += np.outer(context, context)
        self.b[chosen_arm] += actual_reward * context
        self.theta_hat[chosen_arm] = np.linalg.inv(self.A[chosen_arm]).dot(self.b[chosen_arm])

        self.cum[chosen_arm] += np.random.binomial(1, self.cost[chosen_arm])
        self.empirical_cost_means[chosen_arm] = self.cum[chosen_arm] / (self.arm_counts[chosen_arm] + 1)
        self.budget -= self.cost[chosen_arm]

    def run(self):
        """
        Run the LINUCB algorithm over all rounds within the given budget.
        """
        # Calculate true rewards based on context and true_theta
        true_rewards = self.contexts.dot(self.true_theta.T)
        i = 0
        while self.budget > np.max(self.cost):
            context = self.contexts[i]
            chosen_arm = self.select_arm(context, i)
            self.arm_counts[chosen_arm] += 1

            # Calculate reward for chosen arm based on true theta
            actual_reward = true_rewards[i, chosen_arm] / self.cost[chosen_arm]
            optimal_arm = np.argmax(true_rewards[i] / self.cost)

            # Update rewards and norms
            self.rewards[i] = actual_reward
            opt_rew = true_rewards[i, optimal_arm] / self.cost[optimal_arm]
            self.optimal_reward.append(opt_rew)
            self.norms[i] = np.linalg.norm(self.theta_hat - self.true_theta, 'fro')

            # Update parameters for the chosen arm
            self.update_parameters(chosen_arm, context, true_rewards[i, chosen_arm])
            self.choices[i] = chosen_arm

            self.summed_regret += (opt_rew - actual_reward)

            self.logger.track_rep(self.repetition)
            self.logger.track_approach(2)
            self.logger.track_round(i)
            self.logger.track_regret(self.summed_regret)
            self.logger.track_normalized_budget((self.og_budget - self.budget)/ self.og_budget)
            self.logger.track_spent_budget(self.og_budget - self.budget)
            self.logger.finalize_round()
            i += 1

    def plot_results(self):
        """
        Plot the results showing the cumulative reward and convergence of norms.
        """
        plt.figure(figsize=(14, 6))

        # Plot cumulative reward
        plt.subplot(1, 2, 1)
        plt.plot(np.cumsum(self.optimal_reward) - np.cumsum(self.rewards)[:len(self.optimal_reward)], label='Cumulative regret')
        plt.xlabel('Rounds')
        plt.ylabel('Cumulative Reward')
        plt.legend()
        plt.title('Cumulative Reward vs Optimal')

        # Plot norms to check convergence
        plt.subplot(1, 2, 2)
        plt.plot(self.norms, label='||theta_hat - theta||_F')
        plt.xlabel('Rounds')
        plt.ylabel('Norm')
        plt.legend()
        plt.title('Convergence of Theta Estimates')

        plt.show()


# Parameters
n = 2500
k = 3
n_a = 3
contexts = np.random.random((n, k))
true_theta = np.array([[0.5, 0.1, 0.2], [0.1, 0.5, 0.2], [0.2, 0.1, 0.5]])
cost = np.array([0.8, 1, 0.6])
alpha = 0.2
budget = 1000

# Run the LinUCB algorithm
#linucb = LinUCB(n_actions=n_a, n_features=k, contexts=contexts, true_theta=true_theta, cost=cost, alpha=alpha,
             #   budget=budget)
#linucb.run()
#Llinucb.plot_results()