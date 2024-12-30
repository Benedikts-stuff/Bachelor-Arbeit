import numpy as np
from matplotlib import pyplot as plt
import json


class OmegaUCB:
    def __init__(self, n_actions, n_features, contexts, true_theta, cost, budget, seed, logger, repetition, p, cost_kind):
        """
        Initialize the LinUCB instance with parameters.
        logger sollte None defaulted sein
        n_actions: Number of arms (actions).
        n_features: Number of features for each context.
        contexts: Array of context vectors (data points).
        true_theta: True weight matrix (reward parameter) for each arm.
        cost: Cost per arm.
        alpha: Exploration parameter for the upper confidence bound.
        budget: Total budget for playing arms.
        """
        np.random.seed(seed)
        self.n_actions = n_actions
        self.n_features = n_features
        self.contexts = contexts  #- 0.5
        self.true_theta = true_theta
        self.cost = cost
        self.budget = budget
        self.og_budget = budget
        self.cum = np.zeros(self.n_actions)
        self.arm_counts = np.zeros(self.n_actions)

        self.empirical_cost_means = np.random.rand(self.n_actions)
        self.z = 1
        self.p = p #0.95
        self.repetition = repetition
        self.logger = logger
        self.summed_regret = 0
        self.cost_kind = cost_kind
        self.cost_function = None


        # Initialize variables
        self.A = np.array([np.identity(n_features) for _ in range(n_actions)])  # Covariance matrices for each arm
        self.b = np.zeros((n_actions, n_features))  # Linear predictors for each arm
        self.theta_hat = np.zeros((n_actions, n_features))  # Estimated theta for each arm
        self.choices = np.zeros(len(contexts), dtype=int)
        self.rewards = np.zeros(len(contexts))
        self.optimal_reward = []
        self.norms = np.zeros(len(contexts))

        self.plot_data = [[] for _ in range(self.n_actions)]


    def calculate_upper_confidence_bound(self, context, round):
        """
        Calculate the upper confidence bound for a given action and context.
        """
        upper =[]
        for i in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[i])
            theta_hat = A_inv.dot(self.b[i])
            variance = context.dot(A_inv).dot(context)
            mu_r = theta_hat.dot(context)
            #print(f"mean reward OmegaUCB  in runde {round} und arm {i}", mu_r)
            eta = 1
            arm_count = self.arm_counts[i]
            z = np.sqrt(2* self.p* np.log(round + 2))
            if mu_r != 0 and mu_r != 1:
                eta = 1 #

           # print('LOOOL', mu_r )
            A = arm_count + z**2 * eta
            B = 2*arm_count*mu_r + z**2 * eta # eig noch * (M-m) aber das ist hier gleich 1
            C = arm_count* mu_r**2
            x = np.sqrt((B**2 / (4* A**2)) - (C/A))
            omega_r = (B/(2*A)) + x
            upper.append(omega_r)
            self.plot_data[i].append([omega_r, x])
        # Adjust for cost and return estimated reward per cost ratio
        return upper

    def calculate_lower_confidence_bound(self, context, round):
        """
        Calculate the upper confidence bound for a given action and context.
        """
        lower = []
        for i in range(self.n_actions):
            if self.cost_kind == 'bernoulli':
                mu_c = self.empirical_cost_means[i]
            else:
                mu_c = self.cost[i]#np.random.normal(self.cost[i], 0.0001)

            arm_count = self.arm_counts[i]
            eta = 1
            z = np.sqrt(2 * self.p * np.log(round + 2))

            A = arm_count + z**2 * eta
            B = 2 * arm_count * mu_c + z**2 * eta  # eig noch * (M-m) aber das ist hier gleich 1
            C = arm_count * mu_c**2

            omega_c = B / (2 * A) - np.sqrt((B ** 2 / (4 * A ** 2)) - C / A)
            lower.append(np.clip(omega_c, 0.000001, None))
        # Adjust for cost and return estimated reward per cost ratio
        return lower

    def select_arm(self, context, round):
        """
        Select the arm with the highest upper confidence bound, adjusted for cost.
        """
        upper = np.array(self.calculate_upper_confidence_bound(context, round))
        lower = np.array(self.calculate_lower_confidence_bound(context, round))
        ratio = upper/lower
        return np.argmax(ratio)

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

            # Calculate reward and optimal reward
            actual_reward = true_rewards[i, chosen_arm] / self.cost[chosen_arm]
            #print(f"mean rweward OmegaUCB chosen arm in runde {i} und arm {chosen_arm}",true_rewards[i, chosen_arm])
            optimal_arm = np.argmax(true_rewards[i] / self.cost)

            # Update rewards and norms
            self.rewards[i] = actual_reward
            opt_rew = true_rewards[i, optimal_arm] / self.cost[optimal_arm]
            self.optimal_reward.append(opt_rew)
            self.norms[i] = np.linalg.norm(self.theta_hat - self.true_theta, 'fro')

            # Update parameters for the chosen arm
            self.update_parameters(chosen_arm, context, true_rewards[i, chosen_arm])
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
        print('finish lin w ucb')
        # Liste in einer Datei speichern
        with open('../testing/plot_data.json', 'w') as file:
            json.dump(self.plot_data, file)

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

        # Plot norms to check convergence
        plt.subplot(1, 2, 2)
        plt.plot(self.norms, label='||theta_hat - theta||_F')
        plt.xlabel('Rounds')
        plt.ylabel('Norm')
        plt.legend()
        plt.title('Convergence of Theta Estimates')

        plt.show()

    def set_cost_function(self, func):
        self.cost_function = func



# Parameters
n = 25000
k = 3
n_a = 3
contexts = np.random.random((n, k))
true_theta = np.array([[0.5, 0.1, 0.2], [0.1, 0.5, 0.2], [0.2, 0.1, 0.5]])
cost = np.array([1, 1, 1])
alpha = 0.2
budget = 15000
seed = 0
p = 0.95

# Run the LinUCB algorithm
#omega_ucb = OmegaUCB(n_actions=n_a, n_features=k, contexts=contexts, true_theta=true_theta, cost=cost, budget=budget,repetition=seed ,seed=seed,
 #             p= p)
#omega_ucb.run()
#omega_ucb.plot_results()
#print(omega_ucb.theta_hat)