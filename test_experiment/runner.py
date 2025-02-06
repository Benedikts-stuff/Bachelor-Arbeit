import numpy as np
import torch
import random
import multiprocess as mp

from logger import Logger

class Runner:
    def __init__(self, algo_name, bandit_class, bandit_params, reward_function, cost_function, generator, n_features,
                 n_runs=10, b=1000, filename="experiment_logs.csv", bernoulli=False):
        self.algo_name = algo_name
        self.bandit_class = bandit_class
        self.bandit_params = bandit_params
        self.n_runs = n_runs
        self.generator = generator(n_features)
        self.get_reward = reward_function
        self.get_cost = cost_function
        self.B = b
        self.results = []
        self.filename = filename
        self.bernoulli = bernoulli

    def _run_bandit(self, args):
        seed, run_index = args
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        bandit = self.bandit_class(**self.bandit_params)
        rewards = []
        budget = self.B
        round_num = 0
        local_logger = Logger(self.filename)
        cumulative_regret = 0
        gamma = 1e-8

        # Select new weights for this run
        self.get_reward.reinitialize_weights(seed)
        self.get_cost.reinitialize_weights(seed+42)

        while budget > 0:
            context = self.generator.sample()
            action = bandit.select_arm(context, round_num)
            reward = self.get_reward(context, round_num)
            cost = self.get_cost(context, round_num)

            if self.bernoulli:
                bandit.update(np.random.binomial(1, reward[action]), np.random.binomial(1, cost[action]), action, context)
            else:
                bandit.update(reward[action], cost[action], action, context)

            ratio = reward[action]/(cost[action]+gamma)
            optimal_ratio =  np.max(np.array(reward)/(np.array(cost)+gamma))
            regret = np.clip(optimal_ratio - ratio, 0, 1)
            cumulative_regret += regret
            normalized_used_budget = (self.B-budget) / self.B

            local_logger.log(self.algo_name, round_num, reward[action], cumulative_regret, normalized_used_budget, run_index, seed)

            rewards.append(reward[action])
            budget -= cost[action]
            round_num += 1

        local_logger.save_to_csv()  # Speichert alle gesammelten Logs nach dem Run

    def run_experiment(self):
        ctx = mp.get_context('spawn')
        with ctx.Pool(mp.cpu_count()) as pool:
            self.results = pool.map(self._run_bandit, [(i, i) for i in range(self.n_runs)])



