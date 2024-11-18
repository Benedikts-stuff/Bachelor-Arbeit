import math
import numpy as np
from matplotlib import pyplot as plt
from adversary import Adversary


class EXP3:
    def __init__(self, eta, n_round, k_arms, reward_generator, seed):
        np.random.seed(seed)
        self.eta = eta
        self.n_round = n_round
        self.k_arms = k_arms
        self.s_t_hat = np.ones(self.k_arms)
        self.reward_generator = reward_generator
        self.reward_history = []

    def calculate_sampling_distribution(self, arm):
        a = np.exp(self.eta * self.s_t_hat[arm])
        b = 0
        for  i in range(self.k_arms):
            b += np.exp(self.eta * self.s_t_hat[i])

        return a/b

    def sample(self, probs):
        return np.random.choice(self.k_arms, p=probs)

    def update(self, arm, reward, prob):
        for i in range(self.k_arms):
            if i == arm:
                self.s_t_hat[i] += 1  - ((1- reward)/ prob)


    def run(self):
        for i in range(self.n_round):
            p = []
            for arm in range(self.k_arms):
                p.append(self.calculate_sampling_distribution(arm))

            arm = self.sample(p)
            rewards = self.reward_generator.dynamically_changing_rewards(i)

            self.reward_history.append(rewards[arm])
            self.update(arm, rewards[arm], p[arm])



