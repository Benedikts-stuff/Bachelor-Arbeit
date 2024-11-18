import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import pylab as pl
import seaborn as sns
from matplotlib import pyplot as plt
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
from tqdm import tqdm
from utils import*
import time
from bandit_factory import BanditFactory

REP = "rep"
ROUND = r"$t$"
APPROACH = "Approach"
CURRENT_ARM = r"$I_t$"
SPENT_BUDGET = r"spent-budget"
ACTUAL_TOTAL_REWARD = "reward"
REGRET = "Regret"
NORMALIZED_SPENT_BUDGET = "Normalized Budget"
BGREEDY = "B-Greedy"
LINUCB = "Linear BUCB"
OMEGAUCB = "Omega BUCB"
THOMPSON = "Linear TS"

all_ids = [
    REP,
    ROUND,
    APPROACH,
    SPENT_BUDGET,
    ACTUAL_TOTAL_REWARD,
    REGRET,
    NORMALIZED_SPENT_BUDGET,
]

algorithms= [
    BGREEDY,
    LINUCB,
    OMEGAUCB,
    THOMPSON
]



# Runner-Klasse
class Runner:
    def __init__(self, n_rounds, num_arms=3, num_features=3, budget=4000):
        self.iterations = n_rounds
        self.num_arms = num_arms
        self.num_features = num_features
        self.num_rounds = 10000000
        self.context = np.random.rand(self.num_rounds, self.num_features)
        self.budget = budget
        self.normalized_budget_points = np.linspace(0, 1, 100)
        self.epsilon = np.array([0.05, 0.025, 0.1, 0.15, 0.125, 0.075, 0.09, 0.175, 0.2])
        self.p = np.array([0.25, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])

    def run_experiment(self):
        all_data = {name: [] for name in ['EpsilonGreedy', 'ThompsonSampling', 'LinUCB', 'OmegaUCB']}
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = []

            if __name__ == '__main__':
                for i in tqdm(range(self.iterations), desc="Running Bandit Experiments"):
                    true_weights = generate_true_weights(self.num_arms, self.num_features, seed=i)
                    true_cost = generate_true_cost(self.num_arms)

                    for bandit_type in all_data.keys():
                        futures.append(
                            executor.submit(self.run_bandit, bandit_type, true_weights, true_cost, i)
                        )

            for future in futures:
                bandit_type, result = future.result()
                all_data[bandit_type].append(result)

        if __name__ == '__main__':
            plot_data = {name: self.interp_plot(data) for name, data in all_data.items()}
            self.plot_budget_normalised_regret(plot_data)

    def run_bandit(self, bandit_type, true_weights, true_cost, seed):
        np.random.seed(seed)
        bandit = BanditFactory.create(bandit_type, self, true_weights, true_cost, seed)
        bandit.run()
        return bandit_type, bandit.logger.get_dataframe()

    def interp_plot(self, dfs, x_col=NORMALIZED_SPENT_BUDGET, y_col=REGRET):
        axis_list = [
            df[[x_col, y_col]].sort_values(by=x_col).drop_duplicates(x_col).to_numpy() for df in dfs
        ]
        new_axis_xs = np.linspace(0, 1, 100)
        new_axis_ys = [np.interp(new_axis_xs, axis[:, 0], axis[:, 1]) for axis in axis_list]
        midy = np.mean(new_axis_ys, axis=0)
        return pd.DataFrame({x_col: new_axis_xs, y_col: midy})

    def plot_budget_normalised_regret(self, plot_data):
        plt.figure(figsize=(10, 6))
        styles = {
            'EpsilonGreedy': ('blue', '-'),
            'ThompsonSampling': ('green', '--'),
            'LinUCB': ('orange', '-.'),
            'OmegaUCB': ('red', ':')
        }

        for name, df in plot_data.items():
            color, style = styles[name]
            sns.lineplot(x=NORMALIZED_SPENT_BUDGET, y=REGRET, data=df, label=name, color=color, linestyle=style)

        plt.xlabel("Normalized Spent Budget")
        plt.ylabel("Cumulative Regret")
        plt.ylim(0, 500)
        plt.legend()
        plt.title("Regret Comparison of Different Bandit Approaches")
        plt.show()



# Starte dasd Experiment
runner = Runner(30)
start_time = time.time()  # Startzeitpunkt
runner.run_experiment()
end_time = time.time()  # Endzeitpunkt
execution_time = end_time - start_time  # Zeitdifferenz in Sekunden
print(f"Die Methode dauerte {execution_time:.4f} Sekunden.")