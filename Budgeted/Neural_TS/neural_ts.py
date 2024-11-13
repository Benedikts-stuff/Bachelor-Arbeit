'''
This file contains our implementation of the NeuralTS strategy described in https://arxiv.org/abs/2010.00827
The NeuralTS is implemented by analogy to the LinearTS of Emilie Kaufmann used for our lab sessions : http://chercheurs.lille.inria.fr/ekaufman/SDM.html 
'''
import numpy as np
import numpy.linalg as la
import pandas as pd
from matplotlib import pyplot as plt
import time
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split

# library enables to save some interesting vectors/arrays :
import pickle

from BanditGenerator import SinBandit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Neural network for estimating the mean reward of each arm
class MeanEstimator(nn.Module):
    def __init__(self, d, m, L):
        super().__init__()
        self.d = d
        self.m = m
        self.L = L
        assert m % 2 == 0, "m should be pair !"
        assert d % 2 == 0, "d should be pair !"

        self.modules = [nn.Linear(d, m, bias=False), nn.ReLU()] # List die alle Layer enthält, hier wird die 1. layer hinzugefügt und unten die anderen

        for i in range(1, L - 1):
            self.modules.append(nn.Linear(m, m, bias=False))
            self.modules.append(nn.ReLU())

        self.modules.append(nn.Linear(m, 1, bias=False))
        self.modules.append(nn.ReLU())

        self.sequential = nn.ModuleList(self.modules)

    def init_weights(self):
        first_init = np.sqrt(4 / self.m) * torch.randn((self.m, (self.d // 2) - 1)).to(device)
        first_init = torch.cat(
            [first_init, torch.zeros(self.m, 1).to(device), torch.zeros(self.m, 1).to(device), first_init], axis=1)
        self.sequential[0].weight.data = first_init

        for i in range(2, self.L - 1):
            if i % 2 == 0:
                init = np.sqrt(4 / self.m) * torch.randn((self.m, (self.m // 2) - 1)).to(device)
                self.sequential[i].weight.data = torch.cat(
                    [init, torch.zeros(self.m, 1).to(device), torch.zeros(self.m, 1).to(device), init], axis=1)

        last_init = np.sqrt(2 / self.m) * torch.randn(1, self.m // 2).to(device)
        self.sequential[-2].weight.data = torch.cat([last_init, -last_init], axis=1)

    def forward(self, x):
        x = x
        # Pass the input tensor through each of our operations
        for layer in self.sequential:
            x = layer(x)
        return np.sqrt(self.m) * x

## flatten a large tuple containing tensors of different sizes
def flatten(tensor):
    T = torch.tensor([]).to(device)
    for element in tensor:
        T = torch.cat([T, element.to(device).flatten()])
    return T


# concatenation of all the parameters of a NN
def get_theta(model):
    return flatten(model.parameters())


# loss function of the neural TS
def criterion(estimated_reward, reward, m, reg, theta, theta_0):
    return 0.5 * torch.sum(torch.square(estimated_reward - reward)) + 0.5 * m * reg * torch.square(
        torch.norm(theta - theta_0))


# make the transformation of the context vectors so that we met the assumptions of the authors
def transform(x):
    return np.vstack([x / (np.sqrt(2) * la.norm(x)), x / (np.sqrt(2) * la.norm(x))]).reshape(-1)


# generation of context vectors
def generate_contexts(K, d):
    # Generation of normalized features - ||x|| = 1 and x symmetric
    X = torch.randn((K, d // 2))
    X = torch.Tensor(np.array([transform(np.array(x)) for x in X]))
    return X


# NeuralTS algorithm
class NeuralTS:
    """Neural Thompson Sampling Strategy"""

    def __init__(self, X, nu, m, L, reg=1, sigma=None, name='NeuralTS'):
        self.features = X.to(device)
        self.reg = reg
        (self.K, self.d) = X.size()
        self.nu = nu
        self.sigma = sigma
        self.L = L
        self.m = m
        self.strat_name = name
        self.estimator = MeanEstimator(self.d, self.m, self.L)  #Das ist das NN das wir im algo verwenden und updaten
        self.estimator.to(device)
        self.optimizer = torch.optim.Adam(self.estimator.parameters(), lr=10 ** (-4))
        self.current_loss = 0
        self.criterion = criterion
        self.clear()

    def clear(self):
        # initialize the design matrix, its inverse, 
        # the vectors containing the arms chosen and the rewards obtained
        self.t = 1
        self.estimator.init_weights()
        self.theta_zero = get_theta(self.estimator)
        self.p = (self.theta_zero).size(0)
        self.Design = torch.Tensor(self.reg * np.eye(self.p)).to(device)  #Das ist die Matrix U im pseudocode
        self.DesignInv = torch.Tensor((1 / self.reg) * np.eye(self.p)).to(device)
        self.ChosenArms = []
        self.rewards = []

    def chooseArmToPlay(self):
        estimated_rewards = []

        for k in range(self.K):
            f = self.estimator(self.features[k])
            g = torch.autograd.grad(outputs=f, inputs=self.estimator.parameters())
            g = flatten(g).detach()
            start = time.time()
            sigma_squared = (self.reg * (1 / self.m) * torch.matmul(torch.matmul(g.T, self.DesignInv), g)).to(device) #self.reg ist ein Regularisierungsparameter
            sigma = torch.sqrt(sigma_squared)
            r_tilda = (self.nu) * (sigma) * torch.randn(1).to(device) + f.detach()  # ist das selbe wie aus N(f, (v * sigmaˆ2) zu samplen
            estimated_rewards.append(r_tilda.detach().item())

        arm_to_pull = np.argmax(estimated_rewards)
        print('estimated rewards', estimated_rewards)
        self.ChosenArms.append(self.features[arm_to_pull].tolist())

        return arm_to_pull

    def receiveReward(self, arm, reward):

        estimated_rewards = self.estimator(torch.Tensor(self.ChosenArms).to(device))
        self.rewards.append(reward)
        # torch.autograd.set_detect_anomaly(True)
        self.current_loss = self.criterion(estimated_rewards.to(device),
                                           torch.Tensor(torch.Tensor(self.rewards)).to(device), self.m, self.reg,
                                           get_theta(self.estimator), self.theta_zero)

        # gradient descent
        if self.t == 1:
            self.current_loss.backward(retain_graph=True)
        else:
            self.current_loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        # f(x,theta_t)
        f_t = self.estimator(self.features[arm])

        g = torch.autograd.grad(outputs=f_t, inputs=self.estimator.parameters())

        g = flatten(g)
        g = g / (np.sqrt(self.m))

        # online update of the inverse of the design matrix
        #         omega=self.DesignInv@g
        #         self.DesignInv-=(omega@(omega.T))/(1+(g.T)@omega)
        # this was diverging numerically !

        self.Design += torch.matmul(g, g.T).to(device)
        self.DesignInv = torch.inverse(torch.diag(torch.diag(self.Design)))  # approximation proposed by the authors
        self.t += 1

    def update_features(self):
        '''This method simulates the situation where the features are changed at each time t'''
        K=self.K
        d=self.d//2
        # Generation of normalized features - ||x|| = 1 and x symmetric
        X = torch.randn((K, d))
        X = torch.Tensor([transform(x) for x in X])
        self.features = None
        self.features = X.to(device)

    def new_MAB(self):
        '''This method actualize the MAB problem given the new features'''
        bandit = SinBandit(self.features, self.sigma)

        return bandit

    def name(self):
        return self.strat_name


def feature_to_contexts(x, K):
    '''
    Transform a feature vector x into K context vectors x_k
    '''
    d = len(x)
    C = np.zeros(shape=(K, K * d))
    for i in range(K):
        C[i][i * d:(i + 1) * d] = x
    return C.tolist()


# run Neural Thompson sampling
K = 3 # Anzahl Arme
d = 4 # feature dimensions
nu = 1 # exploration variance
m = 10 # width of the NN
L = 10 # Depth of the NN
reg = 1
X = generate_contexts(K, d) # Kontext first round
sigma = 0.2
n_rounds = 50

ts_bandit = NeuralTS(X,nu,m, L, reg, sigma)
ts_bandit.clear()
for i in range(n_rounds):
    arm = ts_bandit.chooseArmToPlay()
    rewards = ts_bandit.new_MAB()
    print('actual rewards', [rewards.arms[i].mean for i in range(K)])
    x = rewards.arms[arm]
    ts_bandit.receiveReward(arm, x.sample())
    ts_bandit.update_features()

