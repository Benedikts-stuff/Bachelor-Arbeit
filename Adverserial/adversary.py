import random

class Adversary:
    def __init__(self, rewards, seed):
        random.seed(seed)
        self.rewards = rewards

    def dynamically_changing_rewards(self, round):
        if round % 100 == 0:
            random.shuffle(self.rewards)

        return self.rewards

