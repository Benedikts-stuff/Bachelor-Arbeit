import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Define the neural network for a single arm
class SingleArmNetwork(nn.Module):
    def __init__(self, context_dim):
        super(SingleArmNetwork, self).__init__()
        self.fc1 = nn.Linear(context_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 1)  # Single reward prediction

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


# Epsilon-Greedy Bandit Algorithm with one network per arm
class EpsilonGreedyBandit:
    def __init__(self, num_arms, num_features, context, true_theta, cost, budget, repetition, logger, seed, epsilon=0.1):
        self.context_dim = num_features
        self.num_actions = num_arms
        self.epsilon = epsilon
        self.context = context
        self.true_theta = true_theta
        self.cost = cost
        self.budget = budget
        self.og_budget = budget
        self.repetition = repetition
        self.logger = logger
        np.random.seed(seed)

        # Create a neural network and optimizer for each arm
        self.models = [SingleArmNetwork(num_features) for _ in range(num_arms)]
        self.optimizers = [optim.Adam(model.parameters(), lr=0.001) for model in self.models]
        self.criterion = nn.MSELoss()

    def select_action(self, context):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)  # Explore
        else:
            context_tensor = torch.FloatTensor(context).unsqueeze(0)
            with torch.no_grad():
                predicted_rewards = [model(context_tensor).item() for model in self.models]
                #print("predicted reward", predicted_rewards)
            return np.argmax(predicted_rewards)  # Exploit

    def update_model(self, contexts, action, rewards):
        batch_size = len(contexts)
        if len(contexts) >= 32:
            batch_size = 32

        dataset = TensorDataset(torch.FloatTensor(contexts),
                                torch.FloatTensor(rewards))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = self.models[action]
        optimizer = self.optimizers[action]

        #context_tensor = torch.FloatTensor(contexts).unsqueeze(0)
        #reward_tensor = torch.FloatTensor([rewards])

        model.train()
        for batch, reward in dataloader:
            optimizer.zero_grad()
            predicted_reward = model(batch).squeeze()
            loss = self.criterion(predicted_reward, reward)
            loss.backward()
            optimizer.step()

    def run(self):
        for t in range(num_rounds):
            # Select a random context
            print(t)
            context = contexts[np.random.randint(num_samples)]

            # Optimal reward for the current context
            optimal_reward = np.max([sigmoid_reward(context, action) for action in range(num_actions)])

            # Select action
            action = bandit.select_action(context)

            # Observe reward
            reward = sigmoid_reward(context, action)

            contexts_seen[action].append(context)
            rewards_observed[action].append(reward)

            # print("actual reward: ",  reward)
            # Calculate regret
            regret = optimal_reward - reward
            total_regret += regret
            cumulative_regret.append(total_regret)

            # Update the model for the selected arm
            bandit.update_model(contexts_seen[action], action, rewards_observed[action])


