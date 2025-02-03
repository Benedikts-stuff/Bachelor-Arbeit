import numpy as np

class Linear:
    def __init__(self, n_arms, n_features):
        self.n_arms = n_arms
        self.n_features = n_features
        self.weights = np.random.rand(n_arms, n_features)
        self.weights /= self.weights.sum(axis=1, keepdims=True)

    def __call__(self, context, round):
        value = np.dot(self.weights, context)
        return value

    def reinitialize_weights(self, seed):
        np.random.seed(seed)
        self.weights = np.random.rand(self.n_arms, self.n_features)
        self.weights /= self.weights.sum(axis=1, keepdims=True)




class AdversarialLinear:
    def __init__(self, n_arms, n_features):
        self.n_arms = n_arms
        self.n_features = n_features
        self.weights = np.random.rand(n_arms, n_features)
        self.weights /= self.weights.sum(axis=1, keepdims=True)

    def __call__(self, context, round):
        if round % 199 == 0: np.random.shuffle(self.weights)
        value = np.dot(self.weights, context)
        return value

    def reinitialize_weights(self, seed):
        np.random.seed(seed)
        self.weights = np.random.rand(self.n_arms, self.n_features)
        self.weights /= self.weights.sum(axis=1, keepdims=True)



class FixedValue:
    def __init__(self, n_arms, n_features):
        self.n_arms = n_arms
        self.n_features = n_features
        self.probabilities = np.random.rand(n_arms)

    def __call__(self, context):
        return self.probabilities

    def reinitialize_weights(self, seed):
        np.random.seed(seed )
        self.probabilities = np.random.rand(self.n_arms)


class StochasticReward:
    def __init__(self, n_arms, n_features):
        self.n_arms = n_arms
        self.n_features = n_features
        self.weights = np.random.rand(n_arms, n_features)
        self.weights /= self.weights.sum(axis=1, keepdims=True)

    def __call__(self, context):
        probability = np.dot(self.weights, context)
        return np.random.binomial(1, probability)

    def reinitialize_weights(self, seed):
        np.random.seed(seed)
        self.weights = np.random.rand(self.n_arms, self.n_features)
        self.weights /= self.weights.sum(axis=1, keepdims=True)

class StochasticCost:
    def __init__(self, n_arms, n_features):
        self.n_arms = n_arms
        self.n_features = n_features
        self.weights = np.random.rand(n_arms, n_features)
        self.weights /= self.weights.sum(axis=1, keepdims=True)

    def __call__(self, context):
        probability = np.dot(self.weights, context)
        return np.random.binomial(1, probability)

    def reinitialize_weights(self, seed):
        np.random.seed(seed + 42)
        self.weights = np.random.rand(self.n_arms, self.n_features)
        self.weights /= self.weights.sum(axis=1, keepdims=True)
