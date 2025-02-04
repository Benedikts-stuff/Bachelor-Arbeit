import numpy as np
from .process_data import PreProcessData


class SyntheticContext:
    def __init__(self, num_features):
        self.num_features = num_features

    def sample(self):
        return np.random.rand(self.num_features)


class FacebookData:
    def __init__(self, num_features):
        pre_processor =  PreProcessData("../test_experiment/data/facebook-ad-campaign-data.csv")
        self.kde = pre_processor.get_kde()

    def sample(self):
        return np.clip(np.array(self.kde.resample(1).T[0], dtype=np.float64),0, None)