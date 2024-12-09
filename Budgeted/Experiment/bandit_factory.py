from Budgeted.c_b_thompson import ThompsonSamplingContextualBandit
from Budgeted.lin_ucb import LinUCB
from Budgeted.olr_e_greedy import EpsilonGreedyContextualBandit
from Budgeted.neural_w_ucb import NeuralOmegaUCB
from Budgeted.w_ucb import OmegaUCB
from logger import BanditLogger
from Budgeted.c_b_thompson_empirical_cost import ThompsonSamplingContextualBanditEmpiric
from Budgeted.gp_ucb import GPUCB
from Budgeted.gp_ts import GPTS
from Budgeted.c_UCB import C_LinUCB
import numpy as np

# Bandit Factory
class BanditFactory:
    @staticmethod
    def create(bandit_type, runner, true_weights, true_cost, seed):
        logger = BanditLogger()
        if bandit_type == 'EpsilonGreedy':
            return EpsilonGreedyContextualBandit(runner.num_features, np.random.choice(runner.epsilon),runner.num_arms, runner.context, true_weights, true_cost,
                          runner.budget, logger, seed, seed)
        elif bandit_type == 'LinearThompsonSampling':
            return ThompsonSamplingContextualBandit(runner.num_features, 1, runner.num_arms, runner.context,
                                                           true_weights, true_cost, runner.budget, logger, seed, seed)
        elif bandit_type == 'C-LinearThompsonSampling':
            return ThompsonSamplingContextualBanditEmpiric(runner.num_features, 1, runner.num_arms, runner.context,
                                                           true_weights, true_cost, runner.budget, logger, seed, seed)
        elif bandit_type == 'LinUCB':
            return LinUCB(runner.num_arms, runner.num_features, runner.context, true_weights, true_cost,
                          runner.budget, logger, seed, seed)
        elif bandit_type == 'LinOmegaUCB':
            return OmegaUCB(runner.num_arms, runner.num_features, runner.context, true_weights, true_cost,
                            runner.budget,  seed, logger, seed, np.random.choice(runner.p))
        elif bandit_type == 'NeuralOmegaUCB':
            return NeuralOmegaUCB(runner.num_arms, runner.num_features, runner.context, true_weights, true_cost,
                            runner.budget,seed, logger, seed, np.random.choice(runner.p))

        elif bandit_type == 'GPUCB':
            return GPUCB(runner.num_arms, runner.num_features, runner.context, true_weights, true_cost,
                            runner.budget,np.random.choice(runner.gamma), seed, logger, seed)
        elif bandit_type == 'GPTS':
            return GPTS(runner.num_arms, runner.num_features, runner.context, true_weights, true_cost,
                            runner.budget,logger, seed, seed)
        elif bandit_type == 'C-LinUCB':
            return C_LinUCB(runner.num_arms, runner.num_features, runner.context, true_weights, true_cost,
                          runner.budget, logger, seed, seed)
        raise ValueError(f"Unknown bandit type: {bandit_type}")