import numpy as np
from matplotlib import pyplot as plt



class Random_Bandit:
    def __init__(self, n_actions, n_features, contexts, true_theta, cost, budget, logger, repetition, seed):
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

        self.empirical_cost_means = np.random.rand(self.n_actions)
        self.repetition = repetition
        self.logger = logger
        self.summed_regret = 0

        self.choices = np.zeros(len(contexts), dtype=int)
        self.rewards = np.zeros(len(contexts))
        self.optimal_reward = []
        self.norms = np.zeros(len(contexts))



    def select_arm(self):
        """
        Select the arm with the highest upper confidence bound, adjusted for cost.
        """
        return np.random.randint(0,2)

    def update_parameters(self, chosen_arm):
        """
        Update the parameters for the chosen arm based on observed context and reward.
        """
        self.budget -= self.cost[chosen_arm]

    def run(self):
        """
        Run the LINUCB algorithm over all rounds within the given budget.
        """
        # Calculate true rewards based on context and true_theta
        true_rewards = self.contexts.dot(self.true_theta.T)
        i = 0
        c = 0
        while self.budget > np.max(self.cost):
            context = self.contexts[i]
            chosen_arm = self.select_arm()
            self.arm_counts[chosen_arm] += 1

            # Calculate reward and optimal reward
            actual_reward = true_rewards[i, chosen_arm] / self.cost[chosen_arm]
            optimal_arm = np.argmax(true_rewards[i] / self.cost)
            if(optimal_arm != chosen_arm):
               c +=1

            # Update rewards and norms
            self.rewards[i] = actual_reward
            opt_rew = true_rewards[i, optimal_arm] / self.cost[optimal_arm]
            self.optimal_reward.append(opt_rew)

            # Update parameters for the chosen arm
            self.update_parameters(chosen_arm)
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
        print('finish random')

