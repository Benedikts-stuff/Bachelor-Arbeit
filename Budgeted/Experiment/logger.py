import numpy as np
import pandas as pd



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

