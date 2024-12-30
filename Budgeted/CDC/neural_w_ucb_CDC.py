import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import time

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




class NeuralOmegaUCB_CDC:
    def __init__(self, n_actions, n_features, contexts, true_theta, cost, budget, repetition, logger, seed, p, cost_kind, reward_func, cost_theta, cost_func):
        """
        Initialize the LinUCB instance with parameters.
        logger sollte None defaulted sein
        n_actions: Number of arms (actions).
        n_features: Number of features for each context.
        contexts: Array of context vectors (data points).
        true_theta: True weight matrix (reward parameter) for each arm.
        cost: Cost per arm.
        alpha: Exploration parameter for the upper confidence bound.
        budget: Total budget for playing arms.
        """
        np.random.seed(seed)
        self.n_actions = n_actions
        self.n_features = n_features
        self.contexts = contexts  #- 0.5
        self.true_theta = true_theta
        self.cost = cost
        self.budget = budget
        self.og_budget = budget
        self.cum = np.zeros(self.n_actions)
        self.arm_counts = np.zeros(self.n_actions)
        self.gamma = 0.00000001
        self.cost_kind = cost_kind
        self.reward_func = reward_func

        self.cost_theta = cost_theta
        self.cost_func = cost_func

        self.empirical_cost_means = np.random.rand(self.n_actions)
        self.z = 1
        self.p = p #0.95
        self.repetition = repetition
        self.logger = logger
        self.summed_regret = 0


        # Initialize variables
        self.models = [SingleArmNetwork(n_features) for _ in range(n_actions)]
        self.optimizer = [optim.Adam(model.parameters(), lr=0.001) for model in self.models]
        self.criterion = nn.MSELoss()

        # models for cost
        self.models_c = [SingleArmNetwork(n_features) for _ in range(n_actions)]
        self.optimizer_c = [optim.Adam(model.parameters(), lr=0.001) for model in self.models_c]
        self.criterion_c = nn.MSELoss()

        self.rewards = np.zeros(len(contexts))
        self.optimal_reward = []

        self.contexts_seen = [[] for _ in range(n_actions)]
        self.rewards_seen =[[] for _ in range(n_actions)]
        self.costs_seen = [[] for _ in range(n_actions)]


    def calculate_upper_confidence_bound(self, context, round):
        """
        Calculate the upper confidence bound for a given action and context.
        """
        context_tensor = torch.FloatTensor(context).unsqueeze(0)
        context_tensor.requires_grad = True
        output = [self.models[t](context_tensor).squeeze(0) for t in range(self.n_actions)]
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
            omega_r = (B/(2*A)) + x
            upper.append(omega_r)

        # Adjust for cost and return estimated reward per cost ratio
        return upper

    def calculate_lower_confidence_bound(self, context,round):
        """
        Calculate the upper confidence bound for a given action and context.
        """
        context_tensor = torch.FloatTensor(context).unsqueeze(0)
        context_tensor.requires_grad = True
        output = [self.models_c[t](context_tensor).squeeze(0) for t in range(self.n_actions)]
        lower = []
        for i, cost in enumerate(output):
            mu_c = cost.item()

            arm_count = self.arm_counts[i]
            eta = 1
            z = np.sqrt(2 * self.p * np.log(round + 2))

            A = arm_count + z**2 * eta
            B = 2 * arm_count * mu_c + z**2 * eta  # eig noch * (M-m) aber das ist hier gleich 1
            C = arm_count * mu_c**2

            omega_c = B / (2 * A) - np.sqrt((B ** 2 / (4 * A ** 2)) - C / A)
            lower.append(np.clip(omega_c, 0.000001, None))
        # Adjust for cost and return estimated reward per cost ratio
        return lower

    def select_arm(self, context, round):
        """
        Select the arm with the highest upper confidence bound, adjusted for cost.
        """
        upper = np.array(self.calculate_upper_confidence_bound(context, round))
        lower = np.array(self.calculate_lower_confidence_bound(context, round))
        ratio = upper/lower
        return np.argmax(ratio)

    def update_parameters(self, action, contexts, rewards):
        """
        Update the parameters for the chosen arm based on observed context and reward.
        """
        batch_size = len(contexts)
        if len(contexts) >= 20:
            batch_size = 20

        dataset = TensorDataset(torch.FloatTensor(contexts),
                                torch.FloatTensor(rewards))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = self.models[action]
        optimizer = self.optimizer[action]

        #context_tensor = torch.FloatTensor(contexts).unsqueeze(0)
        #reward_tensor = torch.FloatTensor([rewards])

        model.train()
        for batch, reward in dataloader:
            optimizer.zero_grad()
            predicted_reward = model(batch).squeeze()
            loss = self.criterion(predicted_reward, reward)
            loss.backward()
            optimizer.step()

    def update_parameters_cost(self, action, contexts, cost):
        """
        Update the parameters for the chosen arm based on observed context and reward.
        """
        batch_size = len(contexts)
        if len(contexts) >= 20:
            batch_size = 20

        dataset = TensorDataset(torch.FloatTensor(contexts),
                                torch.FloatTensor(cost))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model = self.models_c[action]
        optimizer = self.optimizer_c[action]

        # context_tensor = torch.FloatTensor(contexts).unsqueeze(0)
        # reward_tensor = torch.FloatTensor([rewards])

        model.train()
        for batch, cost in dataloader:
            optimizer.zero_grad()
            predicted_reward = model(batch).squeeze()
            loss = self.criterion_c(predicted_reward, cost)
            loss.backward()
            optimizer.step()


    def run(self):
        """
        Run the LINUCB algorithm over all rounds within the given budget.
        """
        # Calculate true rewards based on context and true_theta
        i = 0
        progress = tqdm(total=100000, desc="Processing neural_w_ucb", unit="step", ncols=100, position=None)  # Fortschrittsbalken ohne Total
        while self.budget > np.max(self.cost):
            context = self.contexts[i]
            true_rewards = self.reward_func(context, self.true_theta)
            cost = self.cost_func(context, self.cost_theta)
            chosen_arm = self.select_arm(context, i)
            self.arm_counts[chosen_arm] += 1

            # Calculate reward and optimal reward
            actual_reward = true_rewards[chosen_arm] / cost[chosen_arm]
            #print(f"mean rweward OmegaUCB chosen arm in runde {i} und arm {chosen_arm}", true_rewards[i, chosen_arm])
            optimal_arm = np.argmax(true_rewards / cost)

            # Update rewards and norms
            self.rewards[i] = actual_reward
            opt_rew = true_rewards[optimal_arm] / cost[optimal_arm]
            self.optimal_reward.append(opt_rew)

            # update buffer
            self.contexts_seen[chosen_arm].append(context)
            self.rewards_seen[chosen_arm].append(true_rewards[chosen_arm])
            self.costs_seen[chosen_arm].append(cost[chosen_arm])

            # Update parameters for the chosen arm
            if i < 3000:
                self.update_parameters(chosen_arm, self.contexts_seen[chosen_arm], self.rewards_seen[chosen_arm])
                self.update_parameters_cost(chosen_arm, self.contexts_seen[chosen_arm], self.costs_seen[chosen_arm])


            self.cum[chosen_arm] += np.random.binomial(1, self.cost[chosen_arm])
            self.empirical_cost_means[chosen_arm] = self.cum[chosen_arm] / (self.arm_counts[chosen_arm] + 1)

            self.budget -= cost[chosen_arm]
            self.summed_regret += opt_rew - actual_reward

            self.logger.track_rep(self.repetition)
            self.logger.track_approach(0)
            self.logger.track_round(i)
            self.logger.track_regret(self.summed_regret)
            self.logger.track_normalized_budget((self.og_budget - self.budget)/ self.og_budget)
            self.logger.track_spent_budget(self.og_budget - self.budget)
            self.logger.finalize_round()
            i += 1
            progress.update(1)  # Fortschrittsbalken aktualisieren

        progress.close()
        print('finish neural wucb')