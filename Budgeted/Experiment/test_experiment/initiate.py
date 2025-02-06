from Budgeted.Experiment.test_experiment.bandits.linear.lin_ucb import LinUCB
from Budgeted.Experiment.test_experiment.bandits.linear.lin_c_UCB import cLinCUCB
from Budgeted.Experiment.test_experiment.bandits.linear.fixed_tor_lin_c_ucb import FixedTorLinCUCB
from Budgeted.Experiment.test_experiment.bandits.linear.fixed_lin_ucb import FixedLinUCB
from Budgeted.Experiment.test_experiment.bandits.linear.fixed_lin_ucb_tor import FixedTorLinUCB
from Budgeted.Experiment.test_experiment.bandits.linear.linear_b_thompson_sampling import ThompsonSampling
from Budgeted.Experiment.test_experiment.bandits.linear.lin_ucb_tuned import TunedLinUCB
from Budgeted.Experiment.test_experiment.bandits.exp4 import EXP4
from Budgeted.Experiment.test_experiment.bandits.linear.adverserial_ts import AdvThompsonSampling
from Budgeted.Experiment.test_experiment.bandits.linear.lin_c_UCB_Xia import XiaLinCUCB
from Budgeted.Experiment.test_experiment.bandits.linear.beta_thompson_sampling import BetaThompsonSampling
from Budgeted.Experiment.test_experiment.bandits.linear.epsilon_first import LinFirst
from Budgeted.Experiment.test_experiment.bandits.linear.budget_confidence_ball import BudgetCB
from Budgeted.Experiment.test_experiment.bandits.linear.lin_greedy import LinGreedy
from Budgeted.Experiment.test_experiment.bandits.linear.lin_omega_ucb import LinOmegaUCB
from Budgeted.Experiment.test_experiment.bandits.linear.lin_ucb_tor import TorLinUCB
from Budgeted.Experiment.test_experiment.bandits.linear.lin_c_ucb_tor import TorLinCUCB
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
from Budgeted.Experiment.test_experiment.plot import *
import warnings

# Alle Warnungen ausschalten
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    budget = 10000
    linear1 = [
        ("b-Greedy", LinGreedy, {"context_dim": 5, "n_arms": 3, "epsilon": 4}),
        (r"$\epsilon$-First", LinFirst, {"context_dim": 5, "n_arms": 3, "epsilon": 0.01, "budget": budget}),
        ("c-LinUCB Tor", TorLinCUCB, {"context_dim": 5, "n_arms": 3, "budget": budget}),
        ("c-LinUCB Xia", XiaLinCUCB, {"context_dim": 5, "n_arms": 3, "alpha": 0.25}),
        (r"Lin $\omega$-UCB", LinOmegaUCB, {"context_dim": 5, "n_arms": 3, "p": 0.25}),
        ("Contextual TS", ThompsonSampling, {"context_dim": 5, "n_arms": 3, "v": 0.1}),
        (r"$\beta$-TS", BetaThompsonSampling, {"context_dim": 5, "n_arms": 3, "s": 1}),
        ("Contextual c-TS", CThompsonSampling, {"context_dim": 5, "n_arms": 3, "v": 0.1}),
    ]

    linear2 = [
        ("budget CB", BudgetCB, {"context_dim": 5, "n_arms": 3, "budget": budget}),
        #("Random", Random_Bandit, {"context_dim": 5, "n_arms": 3}),
        ("LinUCB", LinUCB, {"context_dim": 5, "n_arms": 3}),
        #("TunedLinUCB", TunedLinUCB, {"context_dim": 5, "n_arms": 3}),
        ("LinUCB Tor", TorLinUCB, {"context_dim": 5, "n_arms": 3, "budget": budget}),
        ("c-LinUCB", cLinCUCB, {"context_dim": 5, "n_arms": 3}),
        #("AdvThompsonSampling", AdvThompsonSampling, {"context_dim": 5, "n_arms": 3, "v": 0.1}),
    ]

    linear2_optimized = [
        ("budget CB", BudgetCB, {"context_dim": 5, "n_arms": 3, "budget": 12}), #12
        # ("Random", Random_Bandit, {"context_dim": 5, "n_arms": 3}),
        ("LinUCB", FixedLinUCB, {"context_dim": 5, "n_arms": 3, "delta": 0.8}), # 0.8
        # ("TunedLinUCB", TunedLinUCB, {"context_dim": 5, "n_arms": 3}),
        ("LinUCB Tor", FixedTorLinUCB, {"context_dim": 5, "n_arms": 3, "delta": 0.8}),  # 0.8
        ("c-LinUCB", cLinCUCB, {"context_dim": 5, "n_arms": 3, "delta": 2}), #2
    ]

    linear1_optimized = [
        #("b-Greedy", LinGreedy, {"context_dim": 5, "n_arms": 3, "epsilon": 0.001}),
        (r"$\epsilon$-First", LinFirst, {"context_dim": 5, "n_arms": 3, "epsilon": 0.001, "budget": budget}),
        ("c-LinUCB Tor", FixedTorLinCUCB, {"context_dim": 5, "n_arms": 3, "delta": 0.9}),
        ("c-LinUCB Xia", XiaLinCUCB, {"context_dim": 5, "n_arms": 3, "alpha": 0.1}),
        (r"Lin $\omega$-UCB", LinOmegaUCB, {"context_dim": 5, "n_arms": 3, "p": 0.01}),
        #("Contextual TS", ThompsonSampling, {"context_dim": 5, "n_arms": 3, "v": 0.005}),
        #(r"$\beta$-TS", BetaThompsonSampling, {"context_dim": 5, "n_arms": 3, "s": 2000}),
        ("Contextual c-TS", CThompsonSampling, {"context_dim": 5, "n_arms": 3, "v": 0.1}),
    ]

    non_linear = [
        (r"Neural $\omega$-UCB", NeuralOmegaUCB, {"context_dim": 5, "n_arms": 3, "p": 0.25}),
        ("GP UCB", GPUCB, {"context_dim": 5, "n_arms": 3, "gamma": 0.1}),
        (r"GP $\omega$-UCB", GPWUCB, {"context_dim": 5, "n_arms": 3, "p": 0.25}),
        ("GP TS", GPTS, {"context_dim": 5, "n_arms": 3, "delta": 0.1}),
        (r"GP $\beta$-TS", Beta_GPTS, {"context_dim": 5, "n_arms": 3, "delta": 0.1}),
        ("Neural b-Greedy", NeuralGreedy, {"context_dim": 5, "n_arms": 3, "alpha": 4}),

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

    reward_function = NonLinear(**parameter_synthetic)
    cost_function = NonLinear(**parameter_synthetic)
    filename = "experiment_logs.csv"
    context = SyntheticContext
    bernoulli = False

    executor = Executor(non_linear, reward_function, cost_function, context,
                        n_features=5, n_runs=1, b=budget, filename=filename, bernoulli=bernoulli)
    executor.run_all()

    plot_data = interp_plot("experiment_logs.csv")
    color_mapping = create_global_color_mapping(plot_data["algorithm"].unique())
    plot_budget_normalised_regret(plot_data, color_mapping)
    plot_violin_regret(plot_data, color_mapping)