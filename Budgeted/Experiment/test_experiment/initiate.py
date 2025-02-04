from Budgeted.Experiment.test_experiment.bandits.linear.lin_ucb import LinUCB
from Budgeted.Experiment.test_experiment.bandits.linear.lin_c_UCB import LinCUCB
from Budgeted.Experiment.test_experiment.bandits.linear.linear_b_thompson_sampling import ThompsonSampling
from Budgeted.Experiment.test_experiment.bandits.exp4 import EXP4
from Budgeted.Experiment.test_experiment.bandits.random_bandit import Random_Bandit
from Budgeted.Experiment.test_experiment.bandits.linear.adverserial_ts import AdvThompsonSampling
from Budgeted.Experiment.test_experiment.bandits.linear.beta_thompson_sampling import BetaThompsonSampling
from Budgeted.Experiment.test_experiment.bandits.linear.epsilon_first import LinFirst
from Budgeted.Experiment.test_experiment.bandits.linear.budget_confidence_ball import BudgetCB
from Budgeted.Experiment.test_experiment.bandits.linear.c_linear_b_thompson_sampling import CThompsonSampling
from Budgeted.Experiment.test_experiment.bandits.linear.lin_c_UCB_tor import TorLinCUCB
from Budgeted.Experiment.test_experiment.bandits.linear.lin_greedy import LinGreedy
from Budgeted.Experiment.test_experiment.bandits.linear.lin_omega_ucb import LinOmegaUCB
from Budgeted.Experiment.test_experiment.bandits.linear.lin_ucb_tor import TorLinUCB
from Budgeted.Experiment.test_experiment.bandits.linear.c_linear_b_thompson_sampling import CThompsonSampling
from Budgeted.Experiment.test_experiment.bandits.non_linear.beta_gp_thompson_sampling import Beta_GPTS
from Budgeted.Experiment.test_experiment.bandits.non_linear.budgeted_gp_ucb import GPUCB
from Budgeted.Experiment.test_experiment.bandits.non_linear.gp_b_thompson_sampling import GPTS
from Budgeted.Experiment.test_experiment.bandits.non_linear.gp_omega_ucb import GPWUCB
from Budgeted.Experiment.test_experiment.bandits.non_linear.nerual_omega_ucb import NeuralOmegaUCB
from Budgeted.Experiment.test_experiment.bandits.non_linear.neural_greedy import NeuralGreedy
from Budgeted.Experiment.test_experiment.bandits.random_bandit import Random_Bandit

from functions import *
from executor import Executor
from Budgeted.Experiment.test_experiment.data.context import *

if __name__ == "__main__":
    budget = 10000
    linear = [("LinGreedy", LinGreedy, {"context_dim": 5, "n_arms": 3, "epsilon": 4}),
        ("Random", Random_Bandit, {"context_dim": 5, "n_arms": 3}),
        #("LinFirst", LinFirst, {"context_dim": 5, "n_arms": 3, "epsilon": 0.1, "budget": budget}),
        #("LinUCB", LinUCB, {"context_dim": 5, "n_arms": 3}),
        #("LinCUCB", LinCUCB, {"context_dim": 5, "n_arms": 3}),
        #("LinOmegaUCB", LinOmegaUCB, {"context_dim": 5, "n_arms": 3, "p": 0.25}),
        #("ThompsonSampling", ThompsonSampling, {"context_dim": 5, "n_arms": 3, "v": 0.1}),
        #("AdvThompsonSampling", AdvThompsonSampling, {"context_dim": 5, "n_arms": 3, "v": 0.1}),
       # ("betaThompsonSampling", BetaThompsonSampling, {"context_dim": 5, "n_arms": 3}),
        #("budgeted CB", BudgetCB, {"context_dim": 5, "n_arms": 3, "budget": budget}),
        #("CThompsonSampling", CThompsonSampling, {"context_dim": 5, "n_arms": 3, "v": 0.1}),
              ]

    non_linear = [
        #("NeuralOmegaUCB", NeuralOmegaUCB, {"context_dim": 5, "n_arms": 3, "p": 0.25}),
        #("GPUCB", GPUCB, {"context_dim": 5, "n_arms": 3, "gamma": 0.1}),
        #("GP-w-UCB", GPWUCB, {"context_dim": 5, "n_arms": 3, "p": 0.25}),
        #("GPTS", GPTS, {"context_dim": 5, "n_arms": 3, "delta": 0.1}),
        #("Beta_GPTS", Beta_GPTS, {"context_dim": 5, "n_arms": 3, "delta": 0.1}),
        ("NeuralGreedy", NeuralGreedy, {"context_dim": 5, "n_arms": 3, "alpha": 4}),

    ]

    hybrid = [
        ("EXP4", EXP4, {"context_dim": 5, "n_arms": 3, "eta": 0.1, "epsilon": 0.1}),
    ]

    parameter_synthetic = {
        "n_arms": 3,
        "n_features": 5,
        "a": 1,
        "b": 1
    }

    parameter_fb = {
        "n_arms":3
    }

    reward_function = Linear(**parameter_synthetic)
    cost_function = Linear(**parameter_synthetic)
    filename = "experiment_logs.csv"
    context = SyntheticContext
    bernoulli = False

    executor = Executor(linear, reward_function, cost_function, context,
                        n_features=5, n_runs=1, b=budget, filename=filename, bernoulli=bernoulli)
    executor.run_all()