from test_experiment.bandits.linear.fixed_lin_ucb_tor import FixedTorLinUCB
from test_experiment.bandits.exp4 import EXP4
from test_experiment.bandits.non_linear.beta_gp_thompson_sampling import Beta_GPTS
from test_experiment.bandits.non_linear.budgeted_gp_ucb import GPUCB
from test_experiment.bandits.non_linear.gp_b_thompson_sampling import GPTS
from test_experiment.bandits.non_linear.gp_omega_ucb import GPWUCB
from test_experiment.bandits.non_linear.nerual_omega_ucb import NeuralOmegaUCB
from test_experiment.bandits.non_linear.neural_greedy import NeuralGreedy

from functions import *
from executor import Executor
from test_experiment.data.context import *
from test_experiment.plot import *
import os

if __name__ == "__main__":
    budget = 10000

    non_linear = [
        ("NeuralOmegaUCB", NeuralOmegaUCB, {"context_dim": 5, "n_arms": 3, "p": 0.25}),
        ("GPUCB", GPUCB, {"context_dim": 5, "n_arms": 3, "gamma": 0.1}),
        ("GP-w-UCB", GPWUCB, {"context_dim": 5, "n_arms": 3, "p": 0.25}),
        ("GPTS", GPTS, {"context_dim": 5, "n_arms": 3, "delta": 0.1}),
        ("Beta_GPTS", Beta_GPTS, {"context_dim": 5, "n_arms": 3, "delta": 0.1}),
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
    filename = "grid_search_logs.csv"
    context = SyntheticContext
    use_bernoulli = False

    ranges_greedy = [0.001, 0.005, 0.01] # 0.001 oder 1/B
    range_first = [0.7, 0.8, 0.9, 1, 1.2] # 0.001
    ranges_xia = [0.2, 0.4, 0.8,1.0, 1.5, 2.0,3.0,4.0 ] #0.1
    ranges_omega = [0.001,0.005, 0.01, 0.05, 0.1, 0.15, 0.2,0.25, 0.3, 0.35]
    ranges_thompson = [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, ] # 0.01
    ranges_beta_thompson = [ 500, 800, 1000, 2000, 10000]

    param_ranges = {
        #"LinGreedy": ranges_greedy,
        #"LinFirst": range_first,
        #"Fixxed LinCUCB Tor": range_first,
        #"Fixxed LinUCB": range_first,
        #"XiaLinCUCB": ranges_xia,
        #"LinOmegaUCB": ranges_omega,
        #"ThompsonSampling": ranges_thompson,
        #"betaThompsonSampling": ranges_beta_thompson,
        #"CThompsonSampling": ranges_thompson,
        #"budget CB": ranges_xia,
        #"LinUCB": ranges_xia,
        "LinUCB Tor": ranges_xia,
        #"c-LinUCB": ranges_xia,
    }

    logs = {}

    if os.path.exists("results_grid_search.csv"):
        os.remove("results_grid_search.csv")

    if os.path.exists("grid_search_logs.csv"):
        os.remove("grid_search_logs.csv")


    for i in range(8):
        linear = [
            #("LinGreedy", LinGreedy, {"context_dim": 5, "n_arms": 3, "epsilon": ranges_greedy[i]}), #0.001 oder 1/B
            #("LinFirst", LinFirst, {"context_dim": 5, "n_arms": 3, "epsilon": range_first[i], "budget": budget}), # 0.001
            #("XiaLinCUCB", XiaLinCUCB, {"context_dim": 5, "n_arms": 3, "alpha": ranges_xia[i]}), #0.1
            #("LinOmegaUCB", LinOmegaUCB, {"context_dim": 5, "n_arms": 3, "p": ranges_omega[i]}), # 0.01
            #("ThompsonSampling", ThompsonSampling, {"context_dim": 5, "n_arms": 3, "v": ranges_thompson[i]}), # 0.001
            #("betaThompsonSampling", BetaThompsonSampling, {"context_dim": 5, "n_arms": 3, "s": ranges_beta_thompson[i]}), # 2000
            #("CThompsonSampling", CThompsonSampling, {"context_dim": 5, "n_arms": 3, "v": ranges_thompson[i]}), #0.01
            #("Fixxed LinCUCB Tor", FixedTorLinCUCB, {"context_dim": 5, "n_arms": 3, "delta":range_first[i] }), #2
            #("Fixxed LinUCB", FixedTorLinUCB, {"context_dim": 5, "n_arms": 3, "delta": range_first[i]}),  # 0.8

        ]

        linear2_optimized = [
            #("budget CB", BudgetCB, {"context_dim": 5, "n_arms": 3, "budget": ranges_xia[i]}), 12
            # ("Random", Random_Bandit, {"context_dim": 5, "n_arms": 3}),
            #("LinUCB", FixedLinUCB, {"context_dim": 5, "n_arms": 3, "delta": ranges_xia[i]}), # 0.8
            # ("TunedLinUCB", TunedLinUCB, {"context_dim": 5, "n_arms": 3}),
            ("LinUCB Tor", FixedTorLinUCB, {"context_dim": 5, "n_arms": 3, "delta": ranges_xia[i]}), # 0.8
            #("c-LinUCB", cLinCUCB, {"context_dim": 5, "n_arms": 3, "delta": ranges_xia[i]}), #2
            # ("AdvThompsonSampling", AdvThompsonSampling, {"context_dim": 5, "n_arms": 3, "v": 0.1}),
        ]

        if os.path.exists("results_grid_search.csv"):
            os.remove("results_grid_search.csv")

        if os.path.exists("grid_search_logs.csv"):
            os.remove("grid_search_logs.csv")

        executor = Executor(linear2_optimized, reward_function, cost_function, context,
                            n_features=5, n_runs=30, b=budget, filename=filename, bernoulli=use_bernoulli)
        executor.run_all()

        plot_data = interp_plot("grid_search_logs.csv")


        for algorithm, data in plot_data.groupby("algorithm"):
            if algorithm not in logs.keys():
                logs[algorithm]={
                    "algorithm": algorithm,
                    "regret": data["summed"].iloc[0],
                    "parameter": param_ranges.get(algorithm)[i]
                }
            else:
                if logs[algorithm]["regret"] > data["summed"].iloc[0]:
                    logs[algorithm] = {
                        "algorithm": algorithm,
                        "regret": data["summed"].iloc[0],
                        "parameter": param_ranges.get(algorithm)[i]
                    }


    df = pd.DataFrame(logs)

    df.to_csv("results_grid_search.csv")

