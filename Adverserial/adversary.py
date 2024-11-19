import random

class Adversary:
    def __init__(self, rewards, seed):
        random.seed(seed)
        self.rewards = rewards

    def dynamically_changing_rewards(self, round):
        if round % 100 == 0:
            random.shuffle(self.rewards)

        return self.rewards

    def randomly_changing_rewards(self, round):
        if round % 100 == 0:
            self.rewards = [reward + random.uniform(-0.1, 0.1) for reward in self.rewards]
            self.rewards = [min(max(reward, 0), 1) for reward in self.rewards]

        return self.rewards

    def mean_rewards(self, round, max_pulled_arm):
        for i in range(len(self.rewards)):
            if i == max_pulled_arm:
                self.rewards[i] = max(0, self.rewards[i] - 0.1)
            else:
                self.rewards[i] = min(1, self.rewards[i] + 0.05)

        return self.rewards