import numpy as np
np.random.seed(42)
# UCB-Bandit-Algorithmus-Klasse
class UCB_Bandit:
    def __init__(self, n_arms, delta, n_rounds, reward_generator, seed):
        np.random.seed(seed)
        self.n_arms = n_arms
        self.delta = delta
        self.arm_counts = np.zeros(n_arms)
        self.arm_reward_means = np.zeros(n_arms)
        self.n_rounds = n_rounds
        self.reward_generator = reward_generator
        self.reward_history = []

    def select_arm(self):
        if sum(self.arm_counts) == 0:
            return np.random.randint(0, self.n_arms)

        ucb_of_arms = np.full(self.n_arms, np.inf)
        for i in range(self.n_arms):
            if self.arm_counts[i] == 0:
                continue
            else:
                ucb_of_arms[i] = self.arm_reward_means[i] + np.sqrt((2 * np.log(1 / self.delta)) / self.arm_counts[i])

        return np.argmax(ucb_of_arms)

    def update(self, arm, reward):
        self.arm_counts[arm] += 1
        n = self.arm_counts[arm]
        self.arm_reward_means[arm] = ((n - 1) * self.arm_reward_means[arm] + reward) / n

    def run(self):
        for n in range(self.n_rounds):
            arm = self.select_arm()
            rewards = self.reward_generator.dynamically_changing_rewards(n)
            self.update(arm, rewards[arm])
            self.reward_history.append(rewards[arm])



