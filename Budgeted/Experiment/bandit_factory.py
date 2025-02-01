from Budgeted.Experiment.utils import linear_cost_adversary
from Budgeted.c_b_thompson import ThompsonSamplingContextualBandit
from Budgeted.greedy.bbe_greedy import EpsilonGreedyContextualBandit2
from Budgeted.greedy.epsilon_first import EpsilonFirstContextualBandit
from Budgeted.lin_ucb import LinUCB
from Budgeted.greedy.olr_e_greedy import EpsilonGreedyContextualBandit
from Budgeted.CDC.e_greedy_cdc import EGreedy_CDC
from Budgeted.w_ucb.neural_w_ucb import NeuralOmegaUCB
from Budgeted.w_ucb.w_ucb import OmegaUCB
from Budgeted.CDC.w_ucb_cdc import OmegaUCB_CDC
from logger import BanditLogger
from Budgeted.c_b_thompson_empirical_cost import ThompsonSamplingContextualBanditEmpiric
from Budgeted.gp_ucb import GPUCB
from Budgeted.w_ucb.gp_w_ucb import GPWUCB
from Budgeted.gp_ts import GPTS
from Budgeted.c_UCB import C_LinUCB
from Budgeted.random_policy import Random_Bandit
from Budgeted.CDC.neural_w_ucb_CDC import NeuralOmegaUCB_CDC
from Budgeted.CDC.LinUCB_CDC import LinUCB_CDC
from Budgeted.CDC.budget_CB import BudgetCB
from Budgeted.greedy.neural_greedy import NeuralGreedy
import numpy as np

# Bandit Factory
class BanditFactory:
    @staticmethod
    def create(bandit_type, runner, true_weights, true_cost, seed, true_cost_weights, cost_kind, adversary, alpha=0, reward_function=None, cost_function=None):
        print(type(cost_function))
        if adversary:
            reward_function = linear_cost_adversary
            cost_function = linear_cost_adversary

        logger = BanditLogger()
        if bandit_type == 'LinearEpsilonGreedy':
            return EpsilonGreedyContextualBandit(runner.num_features, np.random.choice(runner.epsilon),runner.num_arms, runner.context, true_weights, true_cost,
                          runner.budget, logger, seed, seed, cost_kind)
        if bandit_type == 'EpsilonFirst':
            return EpsilonFirstContextualBandit(runner.num_features, np.random.choice(runner.epsilon),runner.num_arms, runner.context, true_weights, true_cost,
                          runner.budget, logger, seed, seed, cost_kind)
        if bandit_type == 'LinearEpsilonGreedy2':
            return EpsilonGreedyContextualBandit2(runner.num_features, np.random.choice(runner.epsilon),runner.num_arms, runner.context, true_weights, true_cost,
                          runner.budget, logger, seed, seed, cost_kind, alpha)
        elif bandit_type == 'LinearThompsonSampling':
            return ThompsonSamplingContextualBandit(runner.num_features, 1, runner.num_arms, runner.context,
                                                           true_weights, true_cost, runner.budget, logger, seed, seed, cost_kind)
        elif bandit_type == 'RandomBandit':
            return Random_Bandit( runner.num_arms, runner.num_features, runner.context,
                                                           true_weights, true_cost, runner.budget, logger, seed, seed)
        elif bandit_type == 'C-LinearThompsonSampling':
            return ThompsonSamplingContextualBanditEmpiric(runner.num_features, 1, runner.num_arms, runner.context,
                                                           true_weights, true_cost, runner.budget, logger, seed, seed, cost_kind)
        elif bandit_type == 'LinUCB':
            return LinUCB(runner.num_arms, runner.num_features, runner.context, true_weights, true_cost,
                          runner.budget, logger, seed, seed, cost_kind)
        elif bandit_type == 'LinOmegaUCB':
            return OmegaUCB(runner.num_arms, runner.num_features, runner.context, true_weights, true_cost,
                            runner.budget,  seed, logger, seed, np.random.choice(runner.p), cost_kind)
        elif bandit_type == 'NeuralOmegaUCB':
            return NeuralOmegaUCB(runner.num_arms, runner.num_features, runner.context, true_weights, true_cost,
                            runner.budget,seed, logger, seed, np.random.choice(runner.p), cost_kind)
        elif bandit_type == 'NeuralGreedy':
            return NeuralGreedy(np.random.choice(runner.epsilon), runner.num_arms, runner.num_features, runner.context,  true_weights, true_cost,
                            runner.budget,seed, logger, seed, cost_kind)
        elif bandit_type == 'GPUCB':
            return GPUCB(runner.num_arms, runner.num_features, runner.context, true_weights, true_cost,
                            runner.budget,np.random.choice(runner.gamma), seed, logger, seed, cost_kind)
        elif bandit_type == 'GPTS':
            return GPTS(runner.num_arms, runner.num_features, runner.context, true_weights, true_cost,
                            runner.budget,logger, seed, seed, cost_kind)
        elif bandit_type == 'C-LinUCB':
            return C_LinUCB(runner.num_arms, runner.num_features, runner.context, true_weights, true_cost,
                          runner.budget, logger, seed, seed, cost_kind)
        elif bandit_type == 'GPWUCB':
            return GPWUCB(runner.num_arms, runner.num_features, runner.context, true_weights, true_cost,
                            runner.budget,np.random.choice(runner.p), seed, logger, seed, cost_kind)
        elif bandit_type == 'LinOmegaUCB_CDC':
            return OmegaUCB_CDC(runner.num_arms, runner.num_features, runner.context, true_weights, true_cost,
                            runner.budget, seed, logger, seed, np.random.choice(runner.p), true_cost_weights, cost_function, reward_function)
        elif bandit_type == 'LinearEpsilonGreedy_CDC':
            return EGreedy_CDC(runner.num_features, np.random.choice(runner.epsilon),runner.num_arms, runner.context, true_weights, true_cost,
                          runner.budget, logger, seed, seed, true_cost_weights, cost_function, reward_function)
        elif bandit_type == 'NeuralOmegaUCB_CDC':
            return NeuralOmegaUCB_CDC(runner.num_arms, runner.num_features, runner.context, true_weights, true_cost,
                            runner.budget, seed, logger, seed, np.random.choice(runner.p), cost_kind, reward_function, true_cost_weights, cost_function)
        elif bandit_type == 'LinUCB_CDC':
           return LinUCB_CDC(runner.num_arms, runner.num_features, runner.context, true_weights, true_cost,
                            runner.budget, logger, seed, seed, true_cost_weights, reward_function, reward_function)
        elif bandit_type == 'Budget_CB':
            return BudgetCB(runner.num_features, runner.num_arms,runner.budget,1,1, 1, reward_function, cost_function,
                            runner.context, true_weights, true_cost_weights, logger, seed, seed)

        raise ValueError(f"Unknown bandit type: {bandit_type}")