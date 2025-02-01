import numpy as np
from Budgeted.CDC.budget_CB import BudgetCB
from Budgeted.Experiment.utils import generate_true_weights, linear_cost, linear_reward
from Budgeted.Experiment.logger import BanditLogger

# Erzeuge die Werte mit einer Nachkommastelle von 0 bis 5
values = np.arange(0.1, 5.1, 0.1)

# Erzeuge alle möglichen Tupel
combinations = [(x, y) for x in values for y in values]
context = np.random.rand(100000, 3)
summed_regret = 1000000
i = 0
good_comb = (0, 0)
for entry in combinations:
    true_weights = generate_true_weights(3, 3, seed=0)
    true_cost_weights = generate_true_weights(3, 3, seed=42)
    logger = BanditLogger()
    bandit = BudgetCB(entry[0], entry[1], 3, 3, 1000, 1, 1, 1, linear_reward, linear_cost,
                      context, true_weights, true_cost_weights, logger, i, i)
    summed_regret_run = bandit.run()
    print(summed_regret)
    if summed_regret_run <= summed_regret:
        summed_regret = summed_regret_run
        good_comb = entry
    i += 1
print(good_comb, summed_regret)
