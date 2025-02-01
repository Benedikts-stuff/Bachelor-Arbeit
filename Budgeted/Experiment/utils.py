import numpy as np
import seaborn as sns
from seaborn import dark_palette


def normalize_weights(weights):
    """Normalisiert Gewichte pro Zeile auf eine Summe <= 1."""

    for i in range(weights.shape[0]):
        row_sum = np.sum(weights[i])
        if row_sum > 1:
            weights[i] /= row_sum
    return weights

def generate_true_weights(num_arms, num_features,method=None,seed=None):
    """Erzeugt zufällige true_weights."""
    if seed:
        np.random.seed(seed)
    if method == "reward":
        weights = np.random.rand(num_arms, num_features)
        return normalize_weights(weights)
    elif method == "close":
        weights = np.random.normal(loc=0.5, scale=0.05, size=(num_arms, num_features))
        return normalize_weights(weights)
    elif method == "beta":
        weights = np.array([np.random.beta(0.5, 0.5, num_features) for _ in range(num_arms)])
        return normalize_weights(weights)
    else:
        return []


def generate_true_cost(num_arms,cdc, method='uniform'):
    """Erzeugt true_cost für die Banditen."""
    if method == 'uniform':
        return np.random.uniform(0.1, 1, num_arms)
    elif method == 'beta':
        return np.clip(np.random.beta(0.5, 0.5, num_arms), 0.001, 1)

# Reward function
def linear_reward(context, weight):
    return np.clip(np.dot(weight.T, context), 0.000001, None)

def linear_reward_adversary(context, weight, round):
    if round > 1000:
        weight[0], weight[-1] = weight[-1], weight[0]
        return np.clip(np.dot(weight, context), 0.000001, None)

    return np.clip(np.dot(weight, context), 0.000001, None)

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

            rewards.append(0.001*reward/1.7)

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


def polynomial_reward2(context, true_theta):
    rewards = []
    for i in range(len(true_theta)):
        if i == 0:
            #return np.tanh((0.5 * context[0] + 0.3 * context[1] + 0.6 * context[2]))
            reward = (np.exp(-(context[0]-0.5)**2 -(context[1]-0.5)**2) + 0.1)
            rewards.append(reward)

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

# Cost function
def linear_cost_adversary(context, weight, round):
    if round >1000:
        weight[0], weight[-1] = weight[-1], weight[0]
        return np.clip(np.dot(weight, context), 0.000001, None)

    return np.clip(np.dot(weight, context), 0.000001, None)


def polynomial_cost(context, true_theta):
    costs = []
    for arm_id in range(len(true_theta)):
        if arm_id == 0:
            # return np.tanh((0.5 * context[0] + 0.3 * context[1] + 0.6 * context[2]))
            x = context.dot(true_theta[arm_id].T)
            cost = np.sin(10 * x)

            costs.append(np.abs(cost))

        elif arm_id == 1:
            # return np.tanh((0.5 * context[0] + 0.3 * context[1] + 0.6 * context[2]))
            x = context.dot(true_theta[arm_id].T)
            cost = np.sin(10*x)

            costs.append(np.abs(cost))
        elif arm_id == 2:
            # return np.tanh((0.5 * context[0] + 0.3 * context[1] + 0.6 * context[2]))
            x = context.dot(true_theta[arm_id].T)
            cost = np.sin(10*x)
            costs.append(np.abs(cost))

    return costs


# Globales Mapping der Farben erstellen
def create_global_color_mapping(plot_data):
    # Erstelle die Farbpallette basierend auf der Anzahl der Methoden
    n_colors = len(plot_data)
    palette = sns.color_palette("viridis", n_colors=n_colors) #sns.cubehelix_palette(n_colors=n_colors, dark= 0.25, start=2)

    color_mapping = {name: palette[i] for i, name in enumerate(plot_data.keys())}
    return color_mapping