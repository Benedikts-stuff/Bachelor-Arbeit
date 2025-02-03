import numpy as np
from matplotlib import pyplot as plt



class C_LinUCB_CDC:
    def __init__(self, n_actions, n_features, contexts, true_theta, cost, budget, logger, repetition, seed,  true_cost_weights, cost_func, reward_func):
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
        #TODO: hier haben wir ein uniform weighting über ale feature dimensionsen -> ineffizient -> CoFinUCB anschauen
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
        self.gamma = 0.00000001
        self.reward_func = reward_func
        self.cost_func = cost_func
        self.cost_theta = true_cost_weights

        self.empirical_cost_means = np.random.rand(self.n_actions)
        self.repetition = repetition
        self.logger = logger
        self.summed_regret = 0


        # Initialize variables
        self.A = np.array([np.identity(n_features) for _ in range(n_actions)])  # Covariance matrices for each arm
        self.b = np.zeros((n_actions, n_features))  # Linear predictors for each arm
        self.theta_hat = np.zeros((n_actions, n_features))  # Estimated theta for each arm
        self.choices = np.zeros(len(contexts), dtype=int)
        self.rewards_history = np.zeros(len(contexts))

        #cost model
        self.A_c = np.array([np.identity(n_features) for _ in range(n_actions)])  # Covariance matrices for each arm
        self.b_c = np.zeros((n_actions, n_features))  # Linear predictors for each arm
        self.theta_hat_c = np.zeros((n_actions, n_features))  # Estimated theta for each arm
        self.costs_history = np.zeros(len(contexts))

        self.optimal_reward = []
        self.norms = np.zeros(len(contexts))


    def calculate_upper_confidence_bound(self, context, round):
        """
        Calculate the upper confidence bound for a given action and context.
        """
        upper =[]
        for i in range(self.n_actions):
            if self.arm_counts[i]>0:
                A_inv = np.linalg.inv(self.A[i])
                theta_hat = A_inv.dot(self.b[i])
                ta = context.dot(A_inv).dot(context)  # how informative is this?
                #gamma =np.sqrt(np.log(round)/self.arm_counts[i]) #np.log(self.arm_counts[i]/ np.log((round+1)))
                a_upper_ci = (1 + np.sqrt(np.log(2 * (round + 1)) / 2)) * np.sqrt(ta)  # upper part of variance interval

                a_mean = theta_hat.dot(context)  # current estimate of mean
                p = a_mean + a_upper_ci
                upper.append(p)
            else:
                upper.append(100000000)

        # Adjust for cost and return estimated reward per cost ratio
        return upper

    def calculate_lower_confidence_bound(self,context, round):
        """
        Calculate the upper confidence bound for a given action and context.
        """
        lower =[]
        for i in range(self.n_actions):
            if self.arm_counts[i]>0:
                A_inv = np.linalg.inv(self.A_c[i])
                theta_hat = A_inv.dot(self.b_c[i])
                ta = context.dot(A_inv).dot(context)  # how informative is this?
               # gamma =np.sqrt(np.log(round)/self.arm_counts[i]) #np.log(self.arm_counts[i]/ np.log((round+1)))
                a_lower_ci = (1 + np.sqrt(np.log(2 * (round + 1)) / 2)) * np.sqrt(ta)  # upper part of variance interval

                a_mean = theta_hat.dot(context)  # current estimate of mean
                p = a_mean - a_lower_ci
                lower.append(a_mean)
            else:
                lower.append(0.000000001)

        # Adjust for cost and return estimated reward per cost ratio
        return lower

    def select_arm(self, context, round):
        """
        Select the arm with the highest upper confidence bound, adjusted for cost.
        """
        upper = np.array(self.calculate_upper_confidence_bound(context, round))
        lower = np.array(self.calculate_lower_confidence_bound(context,round))
        ratio = upper/(lower)
        return np.argmax(ratio)

    def update_parameters(self, chosen_arm, context, actual_reward, actual_cost):
        """
        Update the parameters for the chosen arm based on observed context and reward.
        """
        self.A[chosen_arm] += np.outer(context, context)
        self.b[chosen_arm] += actual_reward * context
        self.theta_hat[chosen_arm] = np.linalg.inv(self.A[chosen_arm]).dot(self.b[chosen_arm])


        self.A_c[chosen_arm] += np.outer(context, context)
        self.b_c[chosen_arm] += actual_cost * context
        self.theta_hat_c[chosen_arm] = np.linalg.inv(self.A_c[chosen_arm]).dot(self.b_c[chosen_arm])

        #print("true cost weights: ", self.cost_theta, "guesses cost weights: ", self.theta_hat_c)

        self.cum[chosen_arm] += np.random.binomial(1, actual_cost)
        self.empirical_cost_means[chosen_arm] = self.cum[chosen_arm] / (self.arm_counts[chosen_arm] + 1)
        self.budget -= self.cost[chosen_arm]

    def run(self):
        """
        Run the LINUCB algorithm over all rounds within the given budget.
        """
        # Calculate true rewards based on context and true_theta
        i = 0
        while self.budget > np.max(self.cost):
            context = self.contexts[i]
            chosen_arm = self.select_arm(context, i)
            self.arm_counts[chosen_arm] += 1

            # Calculate reward and optimal reward
            true_rewards = self.reward_func(context, self.true_theta, i)
            true_cost = np.clip(self.cost_func(context, self.cost_theta, i), 0, 1)
            actual_reward = true_rewards[chosen_arm] / true_cost[chosen_arm]

            optimal_arm = np.argmax(true_rewards / true_cost)

            # Update rewards and norms
            self.rewards_history[i] = actual_reward
            opt_rew = true_rewards[optimal_arm] / true_cost[optimal_arm]
            self.optimal_reward.append(opt_rew)

            self.norms[i] = np.linalg.norm(self.theta_hat - self.true_theta, 'fro')

            # Update parameters for the chosen arm
            self.update_parameters(chosen_arm, context, true_rewards[chosen_arm], true_cost[chosen_arm])
            self.choices[i] = chosen_arm

            self.summed_regret += opt_rew - actual_reward

            self.logger.track_rep(self.repetition)
            self.logger.track_approach(0)
            self.logger.track_round(i)
            self.logger.track_regret(self.summed_regret)
            self.logger.track_normalized_budget((self.og_budget - self.budget)/ self.og_budget)
            self.logger.track_spent_budget(self.og_budget - self.budget)
            self.logger.finalize_round()
            i += 1
        print('finish linUCB')
