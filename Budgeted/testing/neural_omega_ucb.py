from statistics import covariance
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Define the neural network for reward prediction
class RewardNetwork(nn.Module):
    def __init__(self, context_dim, num_actions):
        super(RewardNetwork, self).__init__()
        self.fc1 = nn.Linear(context_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, num_actions)  # One output per action

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)  # Returns reward estimates for all actions

# Generate synthetic data for a chosen reward function
def generate_data(num_samples, context_dim, num_actions, reward_function):
    np.random.seed(0)
    contexts = np.random.uniform(0, 1, size=(num_samples, context_dim))
    true_rewards = np.zeros((num_samples, num_actions))

    for i in range(num_samples):
        for action in range(num_actions):
            true_rewards[i, action] = reward_function(contexts[i], action)

    return contexts, true_rewards

# Example reward function: sigmoid-based
#def sigmoid_reward(context, action):
    #return 1 / (1 + np.exp(-np.dot(context, np.ones_like(context)) + action * 0.5))

# Wahre Belohnungsfunktion f*
def sigmoid_reward(context, action):
    if action == 0:
        #return np.tanh((0.5 * context[0] + 0.3 * context[1] + 0.6 * context[2]))
        return 1/(1 + np.exp(-(np.tanh(0.5 * context[0] + 0.2 * context[1] + 0.3 * context[2]))))
    elif action == 1:
        return 1 / (1 + np.exp(-np.sin((0.2 * context[0] + 0.5 * context[1] + 0.3 * context[2]))))
        #return np.tanh(0.1 * context[0] + 0.8 * context[1] + 0.1 * context[2])
    elif action == 2:
        return 1 / (1 + np.exp(-np.cos((0.2 * context[0] + 0.3 * context[1] + 0.5 * context[2]))))
        #return np.tanh(0.3 * context[0] + 0.3 * context[1] + 0.6 * context[2])
    elif action == 3:
        return 1 / (1 + np.exp(-(0.2 * context[0] + 0.2 * context[1] + 0.2 * context[2])))
        #return np.tanh(0.2 * context[0] + 0.2 * context[1] + 0.2 * context[2])
    elif action == 4:
        return 1 / (1 + np.exp(-(0.01 * context[0] + 0.4 * context[1] + 0.3 * context[2])))
        #return np.tanh(0.01 * context[0] + 0.4 * context[1] + 0.3 * context[2])

# Epsilon-Greedy Bandit Algorithm
class NeuralWUCBBandit:
    def __init__(self, context_dim, num_actions,p,cost, epsilon=0.1):
        self.context_dim = context_dim
        self.num_actions = num_actions
        self.cost = cost
        self.epsilon = epsilon
        self.model = RewardNetwork(context_dim, num_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.1)
        self.criterion = nn.MSELoss()
        self.U_0 = torch.eye(sum(param.numel() for param in self.model.parameters()))
        self.arm_counts = np.zeros(self.num_actions)
        self.p = p

        self.empirical_cost_means = np.random.rand(self.num_actions)
        self.cumulative_cost = np.ones(self.num_actions)


    def lower_confidence_bound(self, round):
        """
        Calculate the upper confidence bound for a given action and context.
        """
        lower = []
        for i in range(self.num_actions):
            mu_c = self.empirical_cost_means[i]
            arm_count = self.arm_counts[i]
            eta = 1
            z = np.sqrt(2 * self.p * np.log(round + 2))

            A = arm_count + z**2 * eta
            B = 2 * arm_count * mu_c + z**2 * eta  # eig noch * (M-m) aber das ist hier gleich 1
            C = arm_count * mu_c**2

            omega_c = B / (2 * A) - np.sqrt(np.clip((B ** 2 / (4 * A ** 2)) - C / A , 0 , None))
            lower.append(np.clip(omega_c, 0.000001, None))
        # Adjust for cost and return estimated reward per cost ratio
        return lower

    def select_action(self, context, round):
        context_tensor = torch.FloatTensor(context).unsqueeze(0)
        context_tensor.requires_grad = True

        # Vorhersage der Rewards
        output = self.model(context_tensor)
        grad_ucbs = []
        U_inv = torch.linalg.inv(self.U_0)

        update_vectors = []
        output = self.model(context_tensor).squeeze(0)

        for i, reward in enumerate(output):
            # Gradienten für spezifische Aktion berechnen
            #reward = output[0, i]
            #grad = torch.autograd.grad(reward, self.model.parameters(), retain_graph=True, create_graph=True)
            #grad_vector = torch.cat([g.flatten() for g in grad])
            #update_vectors.append(grad_vector)

            #sigma_sq = (grad_vector.T @ U_inv @ grad_vector).item() / 128

            # UCB-Wert berechnen
            mu_r = reward.item()
            arm_count = self.arm_counts[i]
            z = np.sqrt(2 * self.p * np.log(round + 2))
            eta =1 #sigma_sq / 0.25 if 0 < mu_r < 1 else 1

            A = arm_count + z ** 2 * eta
            B = 2 * arm_count * mu_r + z ** 2 * eta
            C = arm_count * mu_r ** 2
            s = np.sqrt(np.clip((B ** 2 / (4 * A ** 2)) - (C / A), 0, None))
            omega_r = (B / (2 * A)) + s
            grad_ucbs.append(omega_r)

        lcb = self.lower_confidence_bound(round)
        arm = np.argmax(grad_ucbs / np.array(lcb))
        #a = update_vectors[arm]
        #self.U_0 += torch.outer(a, a)/ 128
        self.arm_counts[arm] += 1
        return arm

    def update_model(self, contexts, actions, rewards):
        dataset = TensorDataset(torch.FloatTensor(contexts),
                                torch.LongTensor(actions),
                                torch.FloatTensor(rewards))
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        self.model.train()
        for context_batch, action_batch, reward_batch in dataloader:
            self.optimizer.zero_grad()
            predicted_rewards = self.model(context_batch)
            chosen_rewards = predicted_rewards.gather(1, action_batch.unsqueeze(1)).squeeze(1)
            loss = self.criterion(chosen_rewards, reward_batch)
            loss.backward()
            self.optimizer.step()

    def update_cost(self, chosen_arm):
        self.cumulative_cost[chosen_arm] += np.random.binomial(1, self.cost[chosen_arm])
        self.empirical_cost_means[chosen_arm] = self.cumulative_cost[chosen_arm] / (self.arm_counts[chosen_arm] + 1)

# Simulation
# Erweiterte Simulation mit Regret-Berechnung und Plot

def generate_true_cost(num_arms, method='uniform'):
    """Erzeugt true_cost für die Banditen."""
    if method == 'uniform':
        return np.random.uniform(0.1, 1, num_arms)
    elif method == 'beta':
        return np.clip(np.random.beta(0.5, 0.5, num_arms), 0.01, 1)


def run_1(seed, contexts=None, true_rewards=None):
    random.seed(seed)
    torch.manual_seed(seed)
    num_samples = 5000
    context_dim = 5
    num_actions = 3
    epsilon = 0.1
    cost = generate_true_cost(num_actions)
    print("cost", cost)
    budget = 500
    cumulative_cost = 0
    normalized_budget = []

    # Generate data
    #contexts, true_rewards = generate_data(num_samples, context_dim, num_actions, sigmoid_reward)

    # Initialize bandit
    bandit = NeuralWUCBBandit(context_dim, num_actions,0.95,cost, epsilon)

    # Simulation variables
    num_rounds = 1000
    batch_size = 10
    contexts_seen = []
    actions_taken = []
    rewards_observed = []

    cumulative_regret = []
    total_regret = 0

    t= 0
    while budget >= np.max(cost):
        print(t)
        # Select a random context
        context = contexts[np.random.randint(num_samples)]

        # Optimal reward for the current context
        optimal_reward = np.max([sigmoid_reward(context, action) for action in range(num_actions)])

        # Select action
        action = bandit.select_action(context, t)

        # Observe reward
        reward = sigmoid_reward(context, action)

        # Calculate regret
        regret = optimal_reward - reward
        print(regret)
        total_regret += regret
        cumulative_regret.append(total_regret)

        # Store data
        contexts_seen.append(context)
        actions_taken.append(action)
        rewards_observed.append(reward)

        # Update model in batches
        if len(contexts_seen) >= batch_size:
            bandit.update_model(contexts_seen, actions_taken, rewards_observed)
            contexts_seen, actions_taken, rewards_observed = [], [], []

        bandit.update_cost(action)
        budget -= cost[action]
        cumulative_cost += cost[action]
        normalized_budget.append(cumulative_cost/(budget + cumulative_cost))
        print("mpricial cost means", bandit.empirical_cost_means)
        t += 1
    # Plot cumulative regret

    return cumulative_regret, normalized_budget


def run_multiple_runs(run_function, num_runs):
    dfs = []
    num_samples = 5000
    context_dim = 5
    num_actions = 3
    x, r = generate_data(num_samples, context_dim, num_actions, sigmoid_reward)
    for i in range(num_runs):
        print(f"Run {i + 1}/{num_runs}")
        cumulative_regret, remaining_budget = run_function(i, x)

        # Speichere die Ergebnisse in einem DataFrame
        df = pd.DataFrame({
            "normalized_budget": remaining_budget,
            "regret": cumulative_regret
        })
        dfs.append(df)

    return dfs

def interp_plot(dfs, x_col="normalized_budget", y_col="regret"):
    # Erstelle eine Liste der Achsen (normalisiertes Budget und Regret)
    axis_list = [
        df[[x_col, y_col]].sort_values(by=x_col).drop_duplicates(x_col).to_numpy() for df in dfs
    ]

    # Erstelle gleichmäßig verteilte X-Werte für die Interpolation
    new_axis_xs = np.linspace(0, 1, 100)

    # Interpoliere die Y-Werte (Regret) für die neuen X-Werte
    new_axis_ys = [np.interp(new_axis_xs, axis[:, 0], axis[:, 1]) for axis in axis_list]

    # Berechne den Mittelwert der interpolierten Y-Werte
    midy = np.mean(new_axis_ys, axis=0)
    std = np.std(new_axis_ys, axis=0)
    print(std)

    # Erstelle einen DataFrame mit den gemittelten Werten
    return pd.DataFrame({x_col: new_axis_xs, y_col: midy}), std

# Anzahl der Durchläufe
num_runs = 1
start_budget = 300

# Führe den Algorithmus mehrfach aus
dfs= run_multiple_runs(run_1, num_runs)

# Interpoliere und berechne den gemittelten Regret
interp_df, std = interp_plot(dfs, x_col="normalized_budget", y_col="regret")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(interp_df["normalized_budget"], interp_df["regret"], label="Mean Cumulative Regret", color="blue")
plt.fill_between(interp_df["normalized_budget"],interp_df["regret"] - std, interp_df["regret"] + std, color="blue", alpha=0.2, label="Std. Deviation")
plt.xlabel("Normalized Budget")
plt.ylabel(f"Mean Cumulative Regret NEURALOMEGAUCB over {num_runs} runs")
plt.title("Mean Regret vs Normalized Budget")
plt.legend()
plt.grid()
plt.show()