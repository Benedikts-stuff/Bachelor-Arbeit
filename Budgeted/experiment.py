import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import pylab as pl
import seaborn as sns
from matplotlib import pyplot as plt
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing

from Budgeted.c_b_thompson_empirical_cost import ThompsonSamplingContextualBanditEmpiric

multiprocessing.set_start_method("spawn", force=True)
from tqdm import tqdm
import time  # Beispielweise zum Simulieren von Berechnungszeit

from Budgeted.c_b_thompson import ThompsonSamplingContextualBandit
from lin_ucb import LinUCB
from olr_e_greedy import EpsilonGreedyContextualBandit
from w_ucb import OmegaUCB

import time

REP = "rep"
ROUND = r"$t$"
BEST_ARM = "best-arm"
APPROACH = "Approach"
TIME = "time"
K = r"$K$"
CURRENT_ARM = r"$I_t$"
OPTIMAL_TOTAL_REWARD = r"$r_1$"
#OPTIMAL_COST = r"$c_1$"
SPENT_BUDGET = r"spent-budget"
EXPECTED_SPENT_BUDGET = r"expected-spent-budget"
ACTUAL_TOTAL_REWARD = "reward"
AVG_COST_OF_CURRENT_ARM = r"$\mu_i^c$"
AVG_REWARD_OF_CURRENT_ARM = r"$\mu_i^r$"
COST_OF_CURRENT_ARM = r"$c_{i,t}$"
REWARD_OF_CURRENT_ARM = r"$r_{i,t}$"
MINIMUM_AVERAGE_COST = r"$c_{min}$"
REGRET = "Regret"
NORMALIZED_SPENT_BUDGET = "Normalized Budget"
RHO = r"$\rho$"
IS_OUR_APPROACH = "our_approach"
APPROACH_ORDER = "order"
NORMALIZED_REGRET = "Normalized Regret"

BGREEDY = "B-Greedy"
LINUCB = "Linear BUCB"
OMEGAUCB = "Omega BUCB"
THOMPSON = "Linear TS"

all_ids = [
    REP,
    ROUND,
    APPROACH,
#    BEST_ARM,
#    CURRENT_ARM,
#    OPTIMAL_TOTAL_REWARD,
#    OPTIMAL_COST,
    SPENT_BUDGET,
#    EXPECTED_SPENT_BUDGET,
    ACTUAL_TOTAL_REWARD,
#    AVG_COST_OF_CURRENT_ARM,
#    AVG_REWARD_OF_CURRENT_ARM,
#    COST_OF_CURRENT_ARM,
#    REWARD_OF_CURRENT_ARM,
#    MINIMUM_AVERAGE_COST,
    REGRET,
#    TIME,
    NORMALIZED_SPENT_BUDGET,
#   APPROACH_ORDER
]

algorithms= [
    BGREEDY,
    LINUCB,
    OMEGAUCB,
    THOMPSON
]



class BanditLogger:
    """
    Logger class to track the statistics of the bandits during execution
    """

    def __init__(self):
        self._columns = all_ids
        self._data = []
        self._current_row = [np.nan for _ in range(len(self._columns))]
        self._column_indices = {key: i for (i, key) in enumerate(self._columns)}

    def track_approach(self, value: str):
        self._track_value(value, APPROACH)

    def track_round(self, value: int):
        self._track_value(value, ROUND)

    def track_regret(self, value: float):
        self._track_value(value, REGRET)

    def track_normalized_budget(self, value: float):
        self._track_value(value, NORMALIZED_SPENT_BUDGET)

    def track_arm(self, value: int):
        self._track_value(value, CURRENT_ARM)

    def track_best_arm(self, value: int):
        self._track_value(value, BEST_ARM)

    def track_total_reward(self, value):
        self._track_value(value, ACTUAL_TOTAL_REWARD)

    def track_rep(self, value: int):
        self._track_value(value, REP)

    def track_optimal_reward(self, value: float):
        self._track_value(value, OPTIMAL_TOTAL_REWARD)

    def track_reward_sample(self, value: float):
        self._track_value(value, REWARD_OF_CURRENT_ARM)

    def track_spent_budget(self, value: float):
        self._track_value(value, SPENT_BUDGET)

    def finalize_round(self):
        """
        Add the current row to the data that will later become a data frame
        """
        self._data.append(self._current_row)
        self._current_row = [np.nan for _ in range(len(self._columns))]

    def get_dataframe(self) -> pd.DataFrame:
        """
        Creates and returns the data frame based on the currently tracked results and the provided columns
        """
        df = pd.DataFrame(self._data, columns=self._columns)
        self._reset()
        return df

    def _track_value(self, newval, id):
        # Sets the value 'newval' for 'id' in the current row of the data frame
        self._current_row[self._index_of(id)] = newval

    def _index_of(self, id):
        return self._column_indices[id]

    def _reset(self):
        self._current_row = [np.nan for _ in range(len(self._columns))]
        self._data = []





class Runner:

    def __init__(self, n_rounds):
        self.iterations = n_rounds
        self.num_arms = 3
        self.num_features = 3
        self.num_rounds = 10000000
        self.context = np.random.rand(self.num_rounds, self.num_features)
        self.budget = 4000
        self.normalized_budget_points = np.linspace(0, 1, 100)
        self.epsilon = np.array([0.05, 0.025, 0.1, 0.15, 0.125, 0.075, 0.09, 0.175, 0.2])
       #self.delta = np.array([0.1,0.15, 0.2, 0,25, 0.3])
        self.p = np.array([0.25, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])

    def run_experiment(self):
        all_data = {
            #'EpsilonGreedy': [],
            #'ThompsonSampling': [],
            'LinUCB': [],
            'OmegaUCB': []
        }

        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = []
            if __name__ == '__main__':
                for i in tqdm(range(self.iterations)):
                    np.random.seed(i)
                    true_weights = np.random.rand(self.num_arms, self.num_features)
                    for i in range(self.num_arms):
                        row_sum = np.sum(true_weights[i])
                        if row_sum > 1:
                            true_weights[i] /= row_sum

                    true_cost = np.random.uniform(0.1, 1, self.num_arms)
                    #true_cost = np.clip(np.random.beta(0.5, 0.5, self.num_arms), 0.01, 1)
                    # Jede Bandit-Instanz in eine separate Funktion packen und als Future starten
                    #futures.append(executor.submit(self.run_bandit, 'EpsilonGreedy', true_weights, true_cost, i))
                    #futures.append(executor.submit(self.run_bandit, 'ThompsonSampling', true_weights, true_cost, i))
                    futures.append(executor.submit(self.run_bandit, 'LinUCB', true_weights, true_cost, i))
                    futures.append(executor.submit(self.run_bandit, 'OmegaUCB', true_weights, true_cost, i))

            # Ergebnisse sammeln, sobald verfügbar
            for future in futures:
                bandit_type, result = future.result()
                all_data[bandit_type].append(result)

        # Interpolierte Plotdaten berechnen
        if __name__ == '__main__':
            plot_data = {name: self.interp_plot(data) for name, data in all_data.items()}

            self.plot_budget_normalised_regret(plot_data)

    def run_bandit(self, bandit_type, true_weights, true_cost, seed):
        np.random.seed(seed)  # Setze den Seed für Konsistenz
        bandit = None
        logger = BanditLogger()

        if bandit_type == 'EpsilonGreedy':
            bandit = EpsilonGreedyContextualBandit(self.num_features, np.random.choice(self.epsilon),
                                                   self.num_arms, self.context, true_weights, true_cost,
                                                   self.budget, logger, seed, seed)
        elif bandit_type == 'ThompsonSampling':
            bandit = ThompsonSamplingContextualBanditEmpiric(self.num_features, 1, self.num_arms, self.context,
                                                     true_weights, true_cost, self.budget, logger, seed, seed)
        elif bandit_type == 'LinUCB':
            bandit = LinUCB(self.num_arms, self.num_features, self.context, true_weights, true_cost, self.budget,
                            logger, seed, seed)
        elif bandit_type == 'OmegaUCB':
            bandit = OmegaUCB(self.num_arms, self.num_features, self.context, true_weights, true_cost, self.budget,
                              logger, seed, seed, np.random.choice(self.p))

        bandit.run()
        return bandit_type, logger.get_dataframe()

    def interp_plot(self, dfs, x_col=NORMALIZED_SPENT_BUDGET, y_col=REGRET):
        axis_list = []
        for df in dfs:
            data = df[[x_col, y_col]].sort_values(by=x_col).drop_duplicates(x_col).to_numpy()
            axis_list.append(data)

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

# Starte das Experiment
runner = Runner(30)
start_time = time.time()  # Startzeitpunkt
runner.run_experiment()
end_time = time.time()  # Endzeitpunkt
execution_time = end_time - start_time  # Zeitdifferenz in Sekunden
print(f"Die Methode dauerte {execution_time:.4f} Sekunden.")