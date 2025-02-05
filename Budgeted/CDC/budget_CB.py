import numpy as np

class BudgetCB:
    def __init__(self, d, num_arms, budget, rho, L, S, rewards, costs, contexts, true_theta, true_theta_cost, logger, seed, repetition):
        self.d = d  # Dimension of context vectors
        self.num_arms = num_arms
        self.budget = budget
        self.og_budget = budget
        self.rho = rho
        self.delta =1/budget  # Confidence parameter can be set as 1/B where B is budget
        self.L = 3  # Bound on \|context\|_2
        self.S = S  # Bound on \|theta\|_2
        self.C = 0.0001 #0.001 und 0.005 gut #np.log(1/self.delta)
        self.lamda = 0.0001 # is lower bound for the cost of all arms

        self.A = [rho * np.identity(d) for _ in range(num_arms)]
        self.b_r = [np.zeros(d) for _ in range(num_arms)]
        self.b_c = [np.zeros(d) for _ in range(num_arms)]

        self.total_cost = 0
        self.total_reward = 0
        self.summed_regret = 0

        self.rewards = rewards
        self.costs = costs
        self.contexts = contexts
        self.true_theta = true_theta
        self.true_theta_cost = true_theta_cost

        self.logger = logger
        self.repetition = repetition

        np.random.seed(seed)


    def calculate_epsilon(self, t):
        return self.C * np.sqrt(np.clip(
            self.d * np.log(1 + t * self.L ** 2 / (self.rho * self.d)) + 2 * np.log(1 / self.delta) + self.S * np.sqrt(
                self.rho), 0, None))

    def select_arm(self, context, t):
        epsilon = self.calculate_epsilon(t)
        best_arm = None
        max_value = -np.inf

        if t < self.num_arms:
            return t # spiele jeden arm 1 mal mindestens

        values = []
        for i in range(self.num_arms):
            A_inv = np.linalg.inv(self.A[i])
            mu_r = np.dot(A_inv, self.b_r[i])
            mu_c = np.dot(A_inv, self.b_c[i])

            # Richtungsvektor in die entgegengesetzte Richtung von x
            direction = mu_c - context
            norm_direction = direction / np.linalg.norm(direction)
            v_c_min = mu_c + epsilon * norm_direction

            # Richtungsvektor in Richtung von x
            direction = context - mu_r
            norm_direction = direction / np.linalg.norm(direction)
            v_r_max = mu_r + epsilon * norm_direction

            arm_value = min(np.dot(context,v_r_max), 1) / max(np.dot(context, v_c_min), self.lamda)
            values.append(arm_value)

        return np.argmax(np.array(values))

    def update(self, arm, reward, cost, context):
        self.budget -= cost
        self.total_reward += reward

        self.A[arm] += np.outer(context, context)
        self.b_r[arm] += reward * context
        self.b_c[arm] += cost * context

    def run(self):
        t = 1
        while self.budget >= 1:
            context_t = self.contexts[t]
            arm = self.select_arm(context_t, t)

            rewards = self.rewards(context_t, self.true_theta, t)
            costs = self.costs(context_t, self.true_theta_cost, t)


            self.update(arm, rewards[arm], costs[arm], context_t)

            opt_rew = np.max(rewards/costs)
            self.summed_regret += opt_rew - (rewards[arm]/costs[arm])

            self.logger.track_rep(self.repetition)
            self.logger.track_approach(0)
            self.logger.track_round(t)
            self.logger.track_regret(self.summed_regret)
            self.logger.track_normalized_budget((self.og_budget - self.budget)/ self.og_budget)
            self.logger.track_spent_budget(self.og_budget - self.budget)
            self.logger.finalize_round()


            t += 1

        return self.summed_regret

