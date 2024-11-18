import numpy as np



def normalize_weights(weights):
    """Normalisiert Gewichte pro Zeile auf eine Summe <= 1."""
    for i in range(weights.shape[0]):
        row_sum = np.sum(weights[i])
        if row_sum > 1:
            weights[i] /= row_sum
    return weights

def generate_true_weights(num_arms, num_features, seed=None):
    """Erzeugt zufällige true_weights."""
    if seed:
        np.random.seed(seed)
    weights = np.random.rand(num_arms, num_features)
    return normalize_weights(weights)

def generate_true_cost(num_arms, method='uniform'):
    """Erzeugt true_cost für die Banditen."""
    if method == 'uniform':
        return np.random.uniform(0.1, 1, num_arms)
    elif method == 'beta':
        return np.clip(np.random.beta(0.5, 0.5, num_arms), 0.01, 1)


