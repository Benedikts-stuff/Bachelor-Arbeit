import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Wahre Belohnungsfunktion f*
def true_reward_function(context_, arm_id):
    if arm_id == 0:
        #return np.tanh((0.5 * context[0] + 0.3 * context[1] + 0.6 * context[2]))
        return 1/(1 + np.exp(-(0.5 * context_[0] + 0.3 * context_[1] + 0.1 * context_[2])))
    elif arm_id == 1:
        return 1 / (1 + np.exp(-(0.1 * context_[0] + 0.4 * context_[1] + 0.1 * context_[2])))
        #return np.tanh(0.1 * context[0] + 0.8 * context[1] + 0.1 * context[2])
    elif arm_id == 2:
        return 1 / (1 + np.exp(-(0.2 * context_[0] + 0.2 * context_[1] + 0.3 * context_[2])))
        #return np.tanh(0.3 * context[0] + 0.3 * context[1] + 0.6 * context[2])
    elif arm_id == 3:
        return 1 / (1 + np.exp(-(0.2 * context_[0] + 0.2 * context_[1] + 0.2 * context_[2])))
        #return np.tanh(0.2 * context[0] + 0.2 * context[1] + 0.2 * context[2])
    elif arm_id == 4:
        return 1 / (1 + np.exp(-(0.01 * context_[0] + 0.4 * context_[1] + 0.3 * context_[2])))
        #return np.tanh(0.01 * context[0] + 0.4 * context[1] + 0.3 * context[2])

class NeuralNetwork(nn.Module):
    def __init__(self, layers):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x)


class NeuralUCB:
    def __init__(self, n_arms, context_dim, hidden_dim, lambda_, eta_nn, J, L, budget, costs, context, seed, p):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.p = p
        self.n_arms = n_arms
        self.lambda_ = 2
        self.eta_nn = eta_nn
        self.J = J
        self.m=hidden_dim
        self.context_dim = context_dim
        self.models = [self.initialize_model(hidden_dim, L) for _ in range(n_arms)]
        self.budget = budget
        self.costs = costs
        self.context = context
        self.t = 0

        # For tracking
        self.arm_counts = torch.zeros(n_arms)
        self.empirical_cost_means = torch.rand(n_arms)
        self.cum_rewards = torch.zeros(n_arms)
        self.selected_arms = []
        self.observed_rewards = []
        self.optimal_rewards = []

    def initialize_model(self, hidden_dim, L):
        layers = [self.context_dim] + [hidden_dim] * (L - 1) + [1]
        model = NeuralNetwork(layers)
        optimizer = optim.SGD(model.parameters(), lr=self.eta_nn)
        covariance = self.lambda_ * torch.eye(sum(p.numel() for p in model.parameters()))
        theta = torch.cat([p.flatten() for p in model.parameters()])
        return {"model": model, "optimizer": optimizer, "covariance": covariance, "theta": theta}

    def calculate_ucb(self, arm_id, context, round):
        x = context.clone().detach().requires_grad_(True)
        model_info = self.models[arm_id]
        model = model_info["model"]
        covariance = model_info["covariance"]
        output = model(x)
        output.backward()
        mu_r = output.item()
        #print("predicted: ", mu_r)

        grad = torch.cat([p.grad.flatten() for p in model.parameters()])
        std = torch.sqrt(((grad.T @ torch.linalg.inv(covariance) @ grad) * self.lambda_)/self.m)

        eta = 1
        arm_count = self.arm_counts[arm_id]
        z = np.sqrt(2 * self.p * np.log(round + 2))
        if mu_r != 0 and mu_r != 1:
            eta = std/ ((1 - mu_r) * mu_r)

        # print('LOOOL', mu_r )
        A = arm_count + z ** 2 * eta
        B = 2 * arm_count * mu_r + z ** 2 * eta  # eig noch * (M-m) aber das ist hier gleich 1
        C = arm_count * mu_r ** 2
        x = torch.sqrt((B ** 2 / (4 * A ** 2)) - (C / A))
        omega_r =(B / (2 * A)) + x

        return omega_r

    def update_model(self, arm_id, context, reward):
        model_info = self.models[arm_id]
        model, optimizer = model_info["model"], model_info["optimizer"]

        context = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
        reward = torch.tensor([reward], dtype=torch.float32)

        for _ in range(self.J):
            optimizer.zero_grad()
            pred = model(context)
            loss = nn.MSELoss()(pred, reward)
            loss.backward()
            optimizer.step()

    def run(self):
        while self.budget > max(self.costs):
            current_context = self.context[self.t]

            # Calculate UCBs
            ucbs = [self.calculate_ucb(arm, current_context, self.t) for arm in range(self.n_arms)]
            selected_arm = torch.argmax(torch.tensor(ucbs)).item()

            # Get reward
            true_reward = [true_reward_function(current_context, arm) for arm in range(self.n_arms)]
            self.selected_arms.append(selected_arm)
            self.observed_rewards.append(true_reward[selected_arm])
            print("Actual: ", selected_arm, "best: ",  np.argmax(true_reward))
            self.optimal_rewards.append(np.max(true_reward))
            #self.cum_rewards[selected_arm] += true_reward
            self.arm_counts[selected_arm] += 1

            # Update budget and model
            self.budget -= self.costs[selected_arm]
            self.update_model(selected_arm, current_context, true_reward[selected_arm])

            self.t += 1


np.random.seed(42)
# Parameter
n_arms = 3
n_rounds = 10000
n_features = 3
num_points = 150
d = 3  # Context dimension
m = 48  # Neural network width 20
L = 3  # Depth of network 3
lambda_ = [1, 1.5, 2]
cost = np.array([1, 1, 1])
eta = [0.1, 0.05, 0.2]
budget = 500
regret_ucb = np.zeros(499)
p = 0.9
J = 10

context = [np.random.uniform(0, 1, n_features) for i in range(n_rounds)]
context_nn = [torch.tensor(context, dtype=torch.float32) for context in context]

for i in range(10):
    nnucb_bandit = NeuralUCB(n_arms, d, m,1, 0.05, J,L, budget, cost, context_nn, i, p)
    nnucb_bandit.run()
    regret_ucb = np.add(regret_ucb, np.array(nnucb_bandit.optimal_rewards) - np.array(nnucb_bandit.observed_rewards))

regret_ucb = regret_ucb / 1
plt.subplot(122)
plt.plot(regret_ucb.cumsum(), label='linear model')
plt.title("Cumulative regret")
plt.legend()
plt.show()

