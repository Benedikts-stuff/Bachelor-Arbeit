import numpy as np

class BudgetCB:
    def __init__(self, n_arms, context_dim, budget):
        self.d = context_dim
        self.gamma = 1e-8
        self.n_arms = n_arms
        self.rho = 2 # basic = 1
        self.delta = 1/budget #budget  # Confidence parameter can be set as 1/B where B is budget
        self.L = 5  # Bound on \|context\|_2
        self.S = 1  # Bound on \|theta\|_2
        self.C = 0.001 #0.001 und 0.005 gut
        self.lamda = 0.0001 # is lower bound for the cost of all arms

        self.A = [self.rho * np.identity(self.d) for _ in range(self.n_arms)]
        self.b_r = [np.zeros(self.d) for _ in range(self.n_arms)]
        self.b_c = [np.zeros(self.d) for _ in range(self.n_arms)]

    def calculate_epsilon(self, t):
        return self.C * np.sqrt(np.clip(
            self.d * np.log(1 + t * self.L**2 / (self.rho * self.d)) + 2 * np.log(1 / self.delta) + self.S * np.sqrt(
                self.rho), 0, None))

    def select_arm(self, context, t):
        epsilon = self.calculate_epsilon(t)

        if t < self.n_arms:
            return t # spiele jeden arm 1 mal mindestens

        values = []
        for i in range(self.n_arms):
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

    def update(self, reward, cost, arm,context):

        self.A[arm] += np.outer(context, context)
        self.b_r[arm] += reward * context
        self.b_c[arm] += cost * context


