import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import pylab as pl
import seaborn as sns
from matplotlib import pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from concurrent.futures import as_completed
import warnings
warnings.filterwarnings("ignore")

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
    def __init__(self, n_rounds, num_arms=3, num_features=3, budget=1000):
        self.iterations = n_rounds
        self.num_arms = num_arms
        self.num_features = num_features
        self.num_rounds = 10000000
        self.context = np.random.rand(self.num_rounds, self.num_features)
        self.budget = budget
        self.normalized_budget_points = np.linspace(0, 1, 100)
        self.epsilon = np.array([4]) #np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        self.p =np.array([0.95])
        self.gamma= np.array([0.1])

    def run_experiment(self):
        all_data = {name: [] for name in ['C-LinUCB', 'LinOmegaUCB','NeuralOmegaUCB', 'LinUCB', 'GPUCB']} #['C-UCB', 'C-ThompsonSampling', 'OmegaUCB','NeuralOmegaUCB', 'LinUCB', 'EpsilonGreedy', 'ThompsonSampling', 'GPUCB', 'GPTS']}   #{name: [] for name in ['EpsilonGreedy', 'ThompsonSampling', 'LinUCB', 'OmegaUCB']}
        #['LinOmegaUCB','NeuralOmegaUCB', 'LinUCB', 'GPUCB', 'GPTS']
        #['C-LinUCB', 'LinOmegaUCB','NeuralOmegaUCB', 'LinUCB', 'GPUCB']
        #[ 'C-LinearThompsonSampling', 'EpsilonGreedy', 'LinearThompsonSampling', 'GPTS']
        with ProcessPoolExecutor(max_workers=9) as executor:
            futures = []

            if __name__ == '__main__':
                for i in tqdm(range(self.iterations), desc="Running Bandit Experiments"):
                    true_weights = generate_true_weights(self.num_arms, self.num_features, seed=i)
                    true_cost = generate_true_cost(self.num_arms, 'beta')

                    for bandit_type in all_data.keys():
                        futures.append(
                            executor.submit(self.run_bandit, bandit_type, true_weights, true_cost, i)
                        )

            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Futures"):
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
            'EpsilonGreedy': ('#1f77b4', '-'),  # Blau (durchgehend)
            'LinearThompsonSampling': ('#ff7f0e', '--'),  # Orange (gestrichelt)
            'LinUCB': ('#2ca02c', '-.'),  # Grün (gepunktet-gestrichelt)
            'LinOmegaUCB': ('#9467bd', '-'),  # Rot (gepunktet)
            'NeuralOmegaUCB': ('#d62728', ':'),  # Lila (durchgehend)
            'GPUCB': ('#8c564b', '--'),  # Braun (gestrichelt)
            'GPTS': ('#e377c2', '-.'),  # Rosa (gepunktet-gestrichelt)
            'C-LinUCB': ('#7f7f7f', ':'),  # Grau (gepunktet)
            'C-LinearThompsonSampling': ('#bcbd22', '--')  # Gelbgrün (gestrichelt)
        }

        for name, df in plot_data.items():
            color, style = styles[name]
            sns.lineplot(x=NORMALIZED_SPENT_BUDGET, y=REGRET, data=df, label=name, color=color, linestyle=style)

        plt.xlabel("Normalized Spent Budget")
        plt.ylabel("Cumulative Regret")
        plt.ylim(0, 250)
        plt.legend()
        plt.title("Regret Comparison of Different Bandit Approaches")
        plt.savefig("comparison_linear_reward_with_budget")
        plt.show()



# Starte dasd Experiment
runner = Runner(30)
start_time = time.time()  # Startzeitpunkt
runner.run_experiment()
end_time = time.time()  # Endzeitpunkt
execution_time = end_time - start_time  # Zeitdifferenz in Sekunden
if __name__ == '__main__':
    print(f"Die Methode dauerte {execution_time:.4f} Sekunden.")