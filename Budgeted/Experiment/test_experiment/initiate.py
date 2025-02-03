from Budgeted.Experiment.test_experiment.bandits.linear.lin_greedy import LinGreedy
from Budgeted.Experiment.test_experiment.bandits.linear.lin_omega_ucb import LinOmegaUCB
from Budgeted.Experiment.test_experiment.bandits.linear.lin_ucb import LinUCB
from Budgeted.Experiment.test_experiment.bandits.linear.lin_c_UCB import LinCUCB
from Budgeted.Experiment.test_experiment.bandits.linear.budget_confidence_ball import BudgetCB
from Budgeted.Experiment.test_experiment.bandits.linear.linear_b_thompson_sampling import ThompsonSampling
from Budgeted.Experiment.test_experiment.bandits.linear.beta_thompson_sampling import BetaThompsonSampling
from Budgeted.Experiment.test_experiment.bandits.non_linear.nerual_omega_ucb import NeuralOmegaUCB
from Budgeted.Experiment.test_experiment.bandits.non_linear.neural_greedy import NeuralGreedy
from Budgeted.Experiment.test_experiment.bandits.non_linear.budgeted_gp_ucb import GPUCB
from Budgeted.Experiment.test_experiment.bandits.non_linear.gp_omega_ucb import GPWUCB
from Budgeted.Experiment.test_experiment.bandits.non_linear.gp_b_thompson_sampling import GPTS
from Budgeted.Experiment.test_experiment.bandits.non_linear.beta_gp_thompson_sampling import Beta_GPTS
from Budgeted.Experiment.test_experiment.bandits.linear.c_linear_b_thompson_sampling import CThompsonSampling
from functions import *
from executor import Executor
from context import *

if __name__ == "__main__":
    budget = 1000

    algorithms = [
        ("LinGreedy", LinGreedy, {"context_dim": 5, "n_arms": 3, "epsilon": 4}),
        ("LinUCB", LinUCB, {"context_dim": 5, "n_arms": 3}),
        ("LinCUCB", LinCUCB, {"context_dim": 5, "n_arms": 3}),
        #("LinOmegaUCB", LinOmegaUCB, {"context_dim": 5, "n_arms": 3, "p": 0.25}),
        #("NeuralOmegaUCB", NeuralOmegaUCB, {"context_dim": 5, "n_arms": 3, "p": 0.25}),
        #("ThompsonSampling", ThompsonSampling, {"context_dim": 5, "n_arms": 3, "v": 0.1}),
        #("BetaThompsonSampling", BetaThompsonSampling, {"context_dim": 5, "n_arms": 3}),
        #("GPUCB", GPUCB, {"context_dim": 5, "n_arms": 3, "gamma": 0.1}),
        #("budgeted CB", BudgetCB, {"context_dim": 5, "n_arms": 3, "budget": budget}),
        #("GP-w-UCB", GPWUCB, {"context_dim": 5, "n_arms": 3, "p": 0.25}),
        #("GPTS", GPTS, {"context_dim": 5, "n_arms": 3, "delta": 0.1}),
        #("Beta_GPTS", Beta_GPTS, {"context_dim": 5, "n_arms": 3, "delta": 0.1}),
        #("NeuralGreedy", NeuralGreedy, {"context_dim": 5, "n_arms": 3, "alpha": 4}),
        #("CThompsonSampling", CThompsonSampling, {"context_dim": 5, "n_arms": 3, "v": 0.1}),
    ]

    reward_function = FixedReward(3, 5)
    cost_function = FixedCost(3, 5)
    filename = "experiment_logs.csv"

    executor = Executor(algorithms, reward_function, cost_function, SyntheticContext,
                        n_features=5, n_runs=30, b=budget, filename=filename)
    executor.run_all()