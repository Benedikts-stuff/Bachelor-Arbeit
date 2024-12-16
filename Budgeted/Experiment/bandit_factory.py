from Budgeted.c_b_thompson import ThompsonSamplingContextualBandit
from Budgeted.lin_ucb import LinUCB
from Budgeted.olr_e_greedy import EpsilonGreedyContextualBandit
from Budgeted.w_ucb.neural_w_ucb import NeuralOmegaUCB
from Budgeted.w_ucb.w_ucb import OmegaUCB
from Budgeted.w_ucb.w_ucb_cdc import OmegaUCB_CDC
from logger import BanditLogger
from Budgeted.c_b_thompson_empirical_cost import ThompsonSamplingContextualBanditEmpiric
from Budgeted.gp_ucb import GPUCB
from Budgeted.w_ucb.gp_w_ucb import GPWUCB
from Budgeted.gp_ts import GPTS
from Budgeted.c_UCB import C_LinUCB
import numpy as np

# Bandit Factory
class BanditFactory:
    @staticmethod
    def create(bandit_type, runner, true_weights, true_cost, seed, true_cost_weights, cost_kind, reward_function):
        logger = BanditLogger()
        if bandit_type == 'LinearEpsilonGreedy':
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
                            runner.budget,  seed, logger, seed, np.random.choice(runner.p), cost_kind)
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
        elif bandit_type == 'GPWUCB':
            return GPWUCB(runner.num_arms, runner.num_features, runner.context, true_weights, true_cost,
                            runner.budget,np.random.choice(runner.p), seed, logger, seed)
        elif bandit_type == 'LinOmegaUCB_CDC':
            return OmegaUCB_CDC(runner.num_arms, runner.num_features, runner.context, true_weights, true_cost,
                            runner.budget, seed, logger, seed, np.random.choice(runner.p), true_cost_weights, reward_function)

        raise ValueError(f"Unknown bandit type: {bandit_type}")