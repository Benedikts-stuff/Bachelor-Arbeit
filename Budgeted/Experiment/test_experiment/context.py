import numpy as np

class SyntheticContext:
    def __init__(self, num_features):
        self.num_features = num_features

    def sample_uniform(self):
        return np.random.rand(self.num_features)