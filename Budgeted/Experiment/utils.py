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
        return np.clip(np.random.beta(0.5, 0.5, num_arms), 0.001, 1)

# Reward function
def linear_reward(context, true_theta):
    return context.dot(true_theta.T)

def polynomial_reward(context, true_theta):
    rewards = []
    for i in range(len(true_theta)):
        if i == 0:
            #return np.tanh((0.5 * context[0] + 0.3 * context[1] + 0.6 * context[2]))
            x = context.dot(true_theta[i].T)
            reward = (
                    10 * (x ** 4)  # Leading term for 4th degree
                    - 15 * (x ** 3)  # Add negative cubic term
                    + 6 * (x ** 2)  # Add positive quadratic term
                    + 0.5 * x  # Linear term
                    + 0.1  # Constant offset
            )

            rewards.append(reward/1.7)

        elif i == 1:
            #return np.tanh((0.5 * context[0] + 0.3 * context[1] + 0.6 * context[2]))
            x = context.dot(true_theta[i].T)
            reward = (
                    10 * (x ** 4)  # Leading term for 4th degree
                    - 15 * (x ** 3)  # Add negative cubic term
                    + 6 * (x ** 2)  # Add positive quadratic term
                    + 0.5 * x  # Linear term
                    + 0.1  # Constant offset
            )

            rewards.append(reward/1.7)

        elif i == 2:
            #return np.tanh((0.5 * context[0] + 0.3 * context[1] + 0.6 * context[2]))
            x= context.dot(true_theta[i].T)
            reward = (
                    10 * (x ** 4)  # Leading term for 4th degree
                    - 15 * (x ** 3)  # Add negative cubic term
                    + 6 * (x ** 2)  # Add positive quadratic term
                    + 0.5 * x  # Linear term
                    + 0.1  # Constant offset
            )
            rewards.append(reward/1.7)

    return rewards

# Cost function
def linear_cost(context, cost_weight):
    return np.clip(np.dot(cost_weight, context), 0.000001, None)


def polynomial_cost(context, true_theta, arm_id):
    costs = []
    if arm_id == 0:
        # return np.tanh((0.5 * context[0] + 0.3 * context[1] + 0.6 * context[2]))
        x = context.dot(true_theta[arm_id].T)
        cost = (
                10 * (x ** 4)  # Leading term for 4th degree
                - 15 * (x ** 3)  # Add negative cubic term
                + 6 * (x ** 2)  # Add positive quadratic term
                + 0.5 * x  # Linear term
                + 0.1  # Constant offset
        )

        costs.append(cost / 1.7)

    elif arm_id == 1:
        # return np.tanh((0.5 * context[0] + 0.3 * context[1] + 0.6 * context[2]))
        x = context.dot(true_theta[arm_id].T)
        cost = (
                10 * (x ** 4)  # Leading term for 4th degree
                - 15 * (x ** 3)  # Add negative cubic term
                + 6 * (x ** 2)  # Add positive quadratic term
                + 0.5 * x  # Linear term
                + 0.1  # Constant offset
        )

        costs.append(cost / 1.7)
    elif arm_id == 2:
        # return np.tanh((0.5 * context[0] + 0.3 * context[1] + 0.6 * context[2]))
        x = context.dot(true_theta[arm_id].T)
        cost = (
                10 * (x ** 4)  # Leading term for 4th degree
                - 15 * (x ** 3)  # Add negative cubic term
                + 6 * (x ** 2)  # Add positive quadratic term
                + 0.5 * x  # Linear term
                + 0.1  # Constant offset
        )
        return costs.append(cost / 1.7)
