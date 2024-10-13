from array import array

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


df = pd.read_csv('./data_linear.csv')
grouped = df.groupby('campaign_id').agg({
    'ctr': 'sum'
}).reset_index()
ctr = grouped['ctr']/ len(df)
ctr = ctr.to_numpy()
max_ctr = ctr.max()
grouped2= df.groupby(['age', 'gender', 'interest' ])
context_counts = grouped2.size().reset_index(name='group_size')
context_probs = context_counts['group_size'] / len(df)

probs = context_probs.to_numpy()

features = context_counts[[ 'age', 'gender', 'interest' ]].to_numpy()



np.random.seed(0)
d = features.shape[1]
n_a = df['campaign_id'].nunique()  # Anzahl der eindeutigen Kampagnen
k= 3 # number of features
n =1000
idxs = [np.random.choice(len(grouped2), p=probs) for _ in range(n)]
#D = np.random.random((n, k)) - 0.5
D = np.zeros((n, k))
for i in range(n):
    D[i] = features[idxs[i]]

D = D - 0.5
#th = np.random.random((n_a, k)) - 0.5
th = np.array([[ 0.25812542,  0.00331851, -0.32298334], [ 0.33253651,  0.01682478, 0.42691954], [ 0.47180655, 0.17512985, -0.18789989]])
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
    p = p + (np.random.random(len(p)) * 0.000001)
    choices[i] = p.argmax()  # choose the highest, line 11
    arm_count[choices[i]] += 1

    # See what kind of result we get
    #rewards[i] = th[choices[i]].dot(x_i)  # using actual theta to figure out reward
    rewards[i] = th[choices[i]].dot(x_i)
    rewards2[i] = ctr[choices[i]]
    # update the input vector
    A[choices[i]] += np.outer(x_i, x_i)
    b[choices[i]] += rewards[i] * x_i

plt.figure(1, figsize=(10, 5))
plt.subplot(121)
plt.plot(norms)
plt.title("Frobeninus norm of estimated theta vs actual")
plt.show()

regret = (P.max(axis=1) - rewards)
regret2 = (P.max(axis=1) - rewards2)
plt.subplot(122)
plt.plot(regret.cumsum(), label='linear model')
plt.plot(regret2.cumsum(), label='fixed model')
plt.title("Cumulative regret")
plt.legend()
plt.show()

plt.subplot(122)
plt.plot(rewards.cumsum(), label= 'linear_model')
plt.plot(rewards2.cumsum(), label= 'fixed_ctr')
plt.title("Cumulative reward")
plt.legend()
plt.show()