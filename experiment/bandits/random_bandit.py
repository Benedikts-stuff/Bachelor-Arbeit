import numpy as np



class Random_Bandit:
    def __init__(self, n_arms, context_dim):
        self.n_actions =n_arms
        self.n_features =context_dim

    def select_arm(self, context, round):
        return np.random.randint(0,self.n_actions)

    def update(self, reward, cost, chosen_arm, context):
        return



