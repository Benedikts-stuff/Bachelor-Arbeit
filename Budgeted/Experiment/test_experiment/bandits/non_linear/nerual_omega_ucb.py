import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .models.neural import SingleArmNetwork


class NeuralOmegaUCB:
    def __init__(self, n_arms, context_dim, p):
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.arm_counts = np.zeros(self.n_arms)
        self.gamma = 1e-8

        self.z = 1
        self.p = p

        # Initialize variables
        self.models = [SingleArmNetwork(self.context_dim) for _ in range(self.n_arms)]
        self.optimizer = [optim.Adam(model.parameters(), lr=0.001) for model in self.models]
        self.criterion = nn.MSELoss()

        # models for cost
        self.models_c = [SingleArmNetwork(self.context_dim) for _ in range(self.n_arms)]
        self.optimizer_c = [optim.Adam(model.parameters(), lr=0.001) for model in self.models_c]
        self.criterion_c = nn.MSELoss()

        self.contexts_seen = [[] for _ in range(self.n_arms)]
        self.rewards_seen =[[] for _ in range(self.n_arms)]
        self.costs_seen = [[] for _ in range(self.n_arms)]


    def calculate_upper_confidence_bound(self, context, round):
        """
        Calculate the upper confidence bound for a given action and context.
        """
        context_tensor = torch.FloatTensor(context).unsqueeze(0)
        context_tensor.requires_grad = True
        output = [self.models[t](context_tensor).squeeze(0) for t in range(self.n_arms)]
        upper =[]
        for i, reward in enumerate(output):
            mu_r = reward.item()
            #print(f"NeuralOmnegaUCB mu_r in round {round} and arm {i}", mu_r)
            eta = 1
            arm_count = self.arm_counts[i]
            z = np.sqrt(2* self.p* np.log(round + 2))
            if mu_r != 0 and mu_r != 1:
                eta = 1 #

           # print('LOOOL', mu_r )
            A = arm_count + z**2 * eta
            B = 2*arm_count*mu_r + z**2 * eta # eig noch * (M-m) aber das ist hier gleich 1
            C = arm_count* mu_r**2
            x = np.sqrt(np.clip((B**2 / (4* A**2)) - (C/A), 0, None))
            omega_r = np.clip((B/(2*A)) + x, 0, 1)
            upper.append(omega_r)

        return upper

    def calculate_lower_confidence_bound(self, context,round):
        context_tensor = torch.FloatTensor(context).unsqueeze(0)
        context_tensor.requires_grad = True
        output = [self.models_c[t](context_tensor).squeeze(0) for t in range(self.n_arms)]
        lower = []
        for i, cost in enumerate(output):
            mu_c = cost.item()

            arm_count = self.arm_counts[i]
            eta = 1
            z = np.sqrt(2 * self.p * np.log(round + 2))

            A = arm_count + z**2 * eta
            B = 2 * arm_count * mu_c + z**2 * eta  # eig noch * (M-m) aber das ist hier gleich 1
            C = arm_count * mu_c**2

            omega_c = B / (2 * A) - np.sqrt(np.clip(    (B ** 2 / (4 * A ** 2)) - C / A, 0, None)    )
            lower.append(np.clip(omega_c, self.gamma, 1))

        return lower

    def select_arm(self, context, round):
        """
        Select the arm with the highest upper confidence bound, adjusted for cost.
        """
        upper = np.array(self.calculate_upper_confidence_bound(context, round))
        lower = np.array(self.calculate_lower_confidence_bound(context, round))
        ratio = upper/lower
        arm = np.argmax(ratio)
        self.contexts_seen[arm].append(context)
        return arm

    def update(self, actual_reward, actual_cost, chosen_arm, context):
        self.arm_counts[chosen_arm] +=1

        self.rewards_seen[chosen_arm].append(actual_reward)
        self.costs_seen[chosen_arm].append(actual_cost)

        if sum(len(v) for v in self.contexts_seen) < 1000:
            self.update_parameters_reward(chosen_arm, self.contexts_seen[chosen_arm], self.rewards_seen[chosen_arm])

            self.update_parameters_cost(chosen_arm,  self.contexts_seen[chosen_arm], self.costs_seen[chosen_arm])

    def update_parameters_reward(self, action, contexts, rewards):
        batch_size = len(contexts)
        if len(contexts) >= 20:
            batch_size = 20

        dataset = TensorDataset(torch.FloatTensor(contexts),
                                torch.FloatTensor(rewards))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = self.models[action]
        optimizer = self.optimizer[action]

        model.train()
        for batch, reward in dataloader:
            optimizer.zero_grad()
            predicted_reward = model(batch).squeeze()
            loss = self.criterion(predicted_reward, reward)
            loss.backward()
            optimizer.step()

    def update_parameters_cost(self, action, contexts, cost):
        batch_size = len(contexts)
        if len(contexts) >= 20:
            batch_size = 20

        dataset = TensorDataset(torch.FloatTensor(contexts),
                                torch.FloatTensor(cost))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = self.models_c[action]
        optimizer = self.optimizer_c[action]

        model.train()
        for batch, cost in dataloader:
            optimizer.zero_grad()
            predicted_reward = model(batch).squeeze()
            loss = self.criterion_c(predicted_reward, cost)
            loss.backward()
            optimizer.step()
