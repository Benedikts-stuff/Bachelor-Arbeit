import numpy as np

from experiment.data.process_data import PreProcessData


def normalize_weights(weights):
    for i in range(weights.shape[0]):
        row_sum = np.sum(weights[i])
        if row_sum > 1:
            weights[i] /= row_sum
    return weights

class Linear:
    def __init__(self, n_arms, n_features, a, b):
        self.n_arms = n_arms
        self.a = a
        self.b = b
        self.n_features = n_features
        self.weights = np.array([np.random.beta(self.a, self.b, self.n_features) for _ in range(self.n_arms)])
        self.weights = normalize_weights(self.weights)

    def __call__(self, context, round):
        value = np.clip(np.dot(self.weights, context), 1e-8, 1)
        return value

    def reinitialize_weights(self, seed):
        np.random.seed(seed)
        self.weights = np.array([np.random.beta(self.a, self.b, self.n_features) for _ in range(self.n_arms)])
        self.weights = normalize_weights(self.weights)




class NonLinear:
    def __init__(self, n_arms, n_features, a, b):
        self.n_arms = n_arms
        self.a = a
        self.b = b
        self.n_features = n_features
        self.weights =np.random.beta(a, b, (n_arms, n_features))
        self.weights = normalize_weights(self.weights)

    def __call__(self, context, round):
        dot_product = np.dot(self.weights, context)
        value = 0.5 + 0.4 * np.sin(5 * dot_product) - 0.1 * (dot_product ** 2)
        return np.clip(value, 0, 1)

    def reinitialize_weights(self, seed):
        np.random.seed(seed)
        self.weights = np.random.beta(self.a, self.b, (self.n_arms, self.n_features))
        self.weights = normalize_weights(self.weights)




class AdversarialLinear:
    def __init__(self, n_arms, n_features, a, b):
        self.n_arms = n_arms
        self.a = a
        self.b = b
        self.n_features = n_features
        self.weights = np.random.beta(a, b, (n_arms, n_features))
        self.weights = normalize_weights(self.weights)

    def __call__(self, context, round):
        if round % 199 == 0: np.random.shuffle(self.weights)
        value = np.dot(self.weights, context)
        return value

    def reinitialize_weights(self, seed):
        np.random.seed(seed)
        self.weights = np.random.beta(self.a, self.b, (self.n_arms, self.n_features))
        self.weights = normalize_weights(self.weights)



class FixedValue:
    def __init__(self, n_arms, n_features, a, b):
        self.n_arms = n_arms
        self.a = a
        self.b = b
        self.n_features = n_features
        self.probabilities = np.random.beta(self.a, self.b,self.n_arms)

    def __call__(self, context, round):
        return self.probabilities

    def reinitialize_weights(self, seed):
        np.random.seed(seed )
        self.probabilities = np.random.beta(self.a, self.b,self.n_arms)



class Stochastic:
    def __init__(self, n_arms, n_features, a, b):
        self.n_arms = n_arms
        self.a = a
        self.b = b
        self.n_features = n_features
        self.weights = np.random.beta(a,b, (n_arms, n_features))
        self.weights = normalize_weights(self.weights)

    def __call__(self, context, round):
        probability = np.dot(self.weights, context)
        return np.random.binomial(1, probability)

    def reinitialize_weights(self, seed):
        np.random.seed(seed)
        self.weights = np.random.beta(self.a,self.b, (self.n_arms, self.n_features))
        self.weights = normalize_weights(self.weights)


class AddDataReward:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        pre_processor =  PreProcessData("./data/facebook-ad-campaign-data.csv")
        self.models = pre_processor.train_gp_models_reward()

    def __call__(self, context, round):
        value = [np.clip(self.models[arm].predict(context.reshape(1, -1), return_std=False), 1e-8, 0.001)[0] * 1000 for arm in range(self.n_arms)]
        return value

    def reinitialize_weights(self, seed):
        return

class AddDataCost:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        pre_processor =  PreProcessData("./data/facebook-ad-campaign-data.csv")
        self.models = pre_processor.train_gp_models_cost()

    def __call__(self, context, round):
        value = [np.clip(self.models[arm].predict(context.reshape(1, -1), return_std=False), 1e-8, 0.001)[0] * 1000 for arm in range(self.n_arms)]
        return value

    def reinitialize_weights(self, seed):
        return