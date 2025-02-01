import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sympy.physics.units import action
from torch.utils.data import DataLoader, TensorDataset
from Budgeted.Experiment.utils import linear_reward
import matplotlib.pyplot as plt

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


class NeuralGreedy:
    def __init__(self, epsilon, num_arms, num_features, context, true_theta, cost, budget, repetition, logger, seed,  cost_kind):
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
        self.reward_func = linear_reward
        self.total_regret = 0
        self.cumulative_regret_history = []
        self.cost_type = cost_kind
        self.empirical_cost_means = np.random.rand(num_arms)
        self.cumulative_cost = np.zeros(num_arms)
        self.arm_counts = np.zeros(num_arms)
        np.random.seed(seed)

        self.models = [SingleArmNetwork(num_features) for _ in range(num_arms)]
        self.optimizers = [optim.Adam(model.parameters(), lr=0.001) for model in self.models]
        self.criterion = nn.MSELoss()

        self.contexts_seen = [[] for _ in range(num_arms)]
        self.rewards_seen =[[] for _ in range(num_arms)]

    def select_action(self, context, t):
        epsilon = min(1, self.epsilon * (self.num_actions / (t + 1)))
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_actions)  # Explore
        else:
            context_tensor = torch.FloatTensor(context).unsqueeze(0)
            with torch.no_grad():
                predicted_rewards = [model(context_tensor).item() for model in self.models]
            if self.cost_type == 'bernoulli':
                return np.argmax(predicted_rewards/self.empirical_cost_means)
            else:
                return np.argmax(predicted_rewards / np.array(self.cost))


    def update_model(self, contexts, action, rewards):
        batch_size = len(contexts)
        if len(contexts) >= 32:
            batch_size = 32

        dataset = TensorDataset(torch.FloatTensor(contexts),
                                torch.FloatTensor(rewards))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = self.models[action]
        optimizer = self.optimizers[action]

        model.train()
        for batch, reward in dataloader:
            optimizer.zero_grad()
            predicted_reward = model(batch).squeeze()
            loss = self.criterion(predicted_reward, reward)
            loss.backward()
            optimizer.step()

    def run(self):
        t = 0
        while self.budget >= np.max(self.cost):
            print(t)
            context_t = self.context[t]

            rewards = self.reward_func(context_t, self.true_theta)
            optimal_reward = np.max(rewards/self.cost)

            action = self.select_action(context_t, t)
            self.arm_counts[action] += 1

            reward = rewards[action]

            self.contexts_seen[action].append(context_t)
            self.rewards_seen[action].append(reward)

            regret = optimal_reward - (reward/self.cost[action])
            self.total_regret += regret
            self.cumulative_regret_history.append(self.total_regret)

            if t < 1000:
                self.update_model(self.contexts_seen[action], action, self.rewards_seen[action])

            self.cumulative_cost[action] += np.random.binomial(1, self.cost[action])
            self.empirical_cost_means[action] = self.cumulative_cost[action] / (self.arm_counts[action] + 1)
            self.budget -= self.cost[action]

            self.logger.track_rep(self.repetition)
            self.logger.track_approach(0)
            self.logger.track_round(t)
            self.logger.track_regret(self.total_regret)
            self.logger.track_normalized_budget((self.og_budget - self.budget) / self.og_budget)
            self.logger.track_spent_budget(self.og_budget - self.budget)
            self.logger.finalize_round()
            t += 1


