from Budgeted.c_b_thompson import ThompsonSamplingContextualBandit
from Budgeted.lin_ucb import LinUCB
from Budgeted.olr_e_greedy import EpsilonGreedyContextualBandit
from Budgeted.w_ucb import OmegaUCB
from logger import BanditLogger
from Budgeted.c_b_thompson_empirical_cost import ThompsonSamplingContextualBanditEmpiric
import numpy as np

# Bandit Factory
class BanditFactory:
    @staticmethod
    def create(bandit_type, runner, true_weights, true_cost, seed):
        logger = BanditLogger()
        if bandit_type == 'EpsilonGreedy':
            return EpsilonGreedyContextualBandit(runner.num_features, np.random.choice(runner.epsilon),
                                                 runner.num_arms, runner.context, true_weights, true_cost,
                                                 runner.budget, logger, seed, seed)
        elif bandit_type == 'ThompsonSampling':
            return ThompsonSamplingContextualBanditEmpiric(runner.num_features, 1, runner.num_arms, runner.context,
                                                           true_weights, true_cost, runner.budget, logger, seed, seed)
        elif bandit_type == 'LinUCB':
            return LinUCB(runner.num_arms, runner.num_features, runner.context, true_weights, true_cost,
                          runner.budget, logger, seed, seed)
        elif bandit_type == 'OmegaUCB':
            return OmegaUCB(runner.num_arms, runner.num_features, runner.context, true_weights, true_cost,
                            runner.budget, logger, seed, seed, np.random.choice(runner.p))
        raise ValueError(f"Unknown bandit type: {bandit_type}")