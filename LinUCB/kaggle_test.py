
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


df = pd.read_csv('./data_linear.csv')
grouped = df.groupby('campaign_id')['ctr'].mean().reset_index()
means = grouped['ctr'].to_numpy()
grouped2= df.groupby(['age', 'gender', 'interest' ])
context_counts = grouped2.size().reset_index(name='group_size')
context_probs = context_counts['group_size'] / len(df)

probs = context_probs.to_numpy()

features = context_counts[[ 'age', 'gender', 'interest' ]].to_numpy()



np.random.seed(42)
d = features.shape[1]
n_a = df['campaign_id'].nunique()  # Anzahl der eindeutigen Kampagnen
k= 3 # number of features
n =1000
#idxs = [np.random.choice(len(grouped2), p=probs) for _ in range(n)]
#D = np.random.random((n, k)) - 0.5
#D = np.zeros((n, k))
#for i in range(n):
#   D[i] = features[idxs[i]]

#D = D - 0.5
D = np.random.random((n, k)) - 0.5
th = np.random.random((n_a, k)) - 0.5
#th2 = np.array([[ 0.25812542,  0.00331851, -0.32298334], [ 0.33253651,  0.01682478, 0.42691954], [ 0.47180655, 0.17512985, -0.18789989]])
#th = np.array([[-0.58948293,  0.85853578,  0.01412974], [ 0.44559418,  0.09203317,  0.0317637 ],[ 0.4316183 , -0.68225067,  0.95054073]])
eps = 0.2
choices = np.zeros(n, dtype=int)
rewards = np.zeros(n)
rewards2 = np.zeros(n)
explore = np.zeros(n)
norms = np.zeros(n)
arm_count = np.zeros(n_a)
b = np.zeros((n_a,k))
A = np.zeros((n_a, k, k))
for a in range(0, n_a):
    A[a] = np.identity(k)
th_hat = np.zeros((n_a, k))  # our temporary feature vectors, our best current guesses
p = np.zeros(n_a)
alph = 0.2
#P = np.zeros((n, n_a))
P = D.dot(th.T)
# LINUCB, usign disjoint model
# This is all from Algorithm 1, p 664, "A contextual bandit appraoch..." Li, Langford
for i in range(0, n):
    x_i = D[i] #features[idxs[i]] - 0.5 # the current context vector
    #P[i] = x_i.dot(th.T)
    for a in range(0, n_a):
        A_inv = np.linalg.inv(A[a])  # we use it twice so cache it.
        th_hat[a] = A_inv.dot(b[a])  # Line 5
        #print(a, ': ', th_hat[a])
        ta = x_i.dot(A_inv).dot(x_i)  # how informative is this?
        a_upper_ci = alph * np.sqrt(ta)  # upper part of variance interval

        a_mean = th_hat[a].dot(x_i)  # current estimate of mean
        p[a] = a_mean + a_upper_ci
    norms[i] = np.linalg.norm(th_hat - th, 'fro')  # diagnostic, are we converging?
    # Let's hnot be biased with tiebraks, but add in some random noise
    p = p + (np.random.random(len(p)) * 0.1)
    choices[i] = p.argmax()  # choose the highest, line 11
    arm_count[choices[i]] += 1

    # See what kind of result we get
    #rewards[i] = th[choices[i]].dot(x_i)  # using actual theta to figure out reward
    rewards[i] = (th[choices[i]].dot(x_i))
    # update the input vector
    A[choices[i]] += np.outer(x_i, x_i)
    b[choices[i]] += rewards[i] * x_i


class UCB_Bandit:
    def __init__(self, n_arms, delta, seed, means, t_rounds):
        self.seed = seed
        np.random.seed(seed)
        self.n_arms = n_arms  # Anzahl der Arme
        self.delta = delta  # Parameter für den Konfidenzbonus
        self.arm_counts = np.zeros(n_arms)  # Zählungen für jeden Arm
        self.arm_reward_means = np.zeros(n_arms)  # Durchschnittliche Belohnung für jeden Arm
        self.actual_means = means
        self.t_rounds = t_rounds
        self.reward_history = np.zeros(self.t_rounds)


    def select_arm(self):
        ucb_of_arms = np.full(self.n_arms, np.inf)
        for i in range(self.n_arms):
            if self.arm_counts[i] == 0:
                continue
            else:
                ucb_of_arms[i] = self.arm_reward_means[i] + np.sqrt((2 * np.log(1 / self.delta)) / self.arm_counts[i])

        return np.argmax(ucb_of_arms)

    def update(self, arm, reward):
        self.arm_counts[arm] += 1
        n = self.arm_counts[arm]
        self.arm_reward_means[arm] = ((n - 1) * self.arm_reward_means[arm] + reward) / n

    def execute(self):
        for n in range(self.t_rounds):
            arm = self.select_arm()
            #print(arm)
            ctr = self.actual_means[arm]
            # reward  = np.random.binomial(1, ctr)
            reward = ctr
            self.update(arm, reward)
            self.reward_history[n] = reward

ucb_bandit = UCB_Bandit(3, 0.5, 42 , means, 1000)
ucb_bandit.execute()
max_mean = means.max()
reward_all = np.cumsum(ucb_bandit.reward_history)
opt = [max_mean for _ in range(ucb_bandit.t_rounds)]
cumulative_optimal_reward = np.cumsum(opt)
cumulative_regret_UCB = cumulative_optimal_reward - reward_all

plt.figure(1, figsize=(10, 5))
plt.subplot(121)
plt.plot(norms)
plt.title("Frobeninus norm of estimated theta vs actual")
plt.show()

regret = (P.max(axis=1) - rewards)
plt.subplot(122)
plt.plot(regret.cumsum(), label='linear model')
#plt.plot(cumulative_regret_UCB, label='ucb')
plt.title("Cumulative regret")
plt.legend()
plt.show()

plt.subplot(122)
plt.plot(rewards.cumsum(), label= 'linear_model')
plt.title("Cumulative reward")
plt.legend()
plt.show()