import numpy as np
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
        return 0.5 * context[0] + 0.3 * context[1] + 0.1 * context[2]
    elif action == 1:
        return 0.1 * context[0] + 0.8 * context[1] + 0.1 * context[2]
        #return np.tanh(0.1 * context[0] + 0.8 * context[1] + 0.1 * context[2])
    elif action == 2:
        return 0.2 * context[0] + 0.2 * context[1] + 0.6 * context[2]
        #return np.tanh(0.3 * context[0] + 0.3 * context[1] + 0.6 * context[2])
    elif action == 3:
        return 0.2 * context[0] + 0.2 * context[1] + 0.2 * context[2]
        #return np.tanh(0.2 * context[0] + 0.2 * context[1] + 0.2 * context[2])
    elif action == 4:
        return 0.01 * context[0] + 0.4 * context[1] + 0.3 * context[2]
        #return np.tanh(0.01 * context[0] + 0.4 * context[1] + 0.3 * context[2])

# Epsilon-Greedy Bandit Algorithm
class EpsilonGreedyBandit:
    def __init__(self, context_dim, num_actions, epsilon=0.1):
        self.context_dim = context_dim
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.model = RewardNetwork(context_dim, num_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def select_action(self, context):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)  # Explore
        else:
            context_tensor = torch.FloatTensor(context).unsqueeze(0)
            with torch.no_grad():
                predicted_rewards = self.model(context_tensor)
            return torch.argmax(predicted_rewards).item()  # Exploit

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

# Simulation
# Erweiterte Simulation mit Regret-Berechnung und Plot

def run_1(seed, contexts, true_rewards):
    np.random.seed(seed)
    num_samples = 20000
    context_dim = 5
    num_actions = 3
    epsilon = 0.1

    # Generate data
    contexts, true_rewards = generate_data(num_samples, context_dim, num_actions, sigmoid_reward)

    # Initialize bandit
    bandit = EpsilonGreedyBandit(context_dim, num_actions, epsilon)

    # Simulation variables
    num_rounds = 20000
    batch_size = 100
    contexts_seen = []
    actions_taken = []
    rewards_observed = []

    cumulative_regret = []
    total_regret = 0

    for t in range(num_rounds):
        # Select a random context
        context = contexts[np.random.randint(num_samples)]

        # Optimal reward for the current context
        optimal_reward = np.max([sigmoid_reward(context, action) for action in range(num_actions)])

        # Select action
        action = bandit.select_action(context)

        # Observe reward
        reward = sigmoid_reward(context, action)

        # Calculate regret
        regret = optimal_reward - reward
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

    # Plot cumulative regret

    return cumulative_regret