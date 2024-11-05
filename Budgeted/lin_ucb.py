import numpy as np
from matplotlib import pyplot as plt


np.random.seed(42)
n = 2500 # number of data points
k = 3 # number of features
n_a = 3 # number of actions
D = np.random.random((n, k)) - 0.5 # our data, or these are the contexts, there are n contexts, each has k features
th = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.2], [0.5, 0.2, 0.3]])# our real theta, what we will try to guess (there are 8 arms, and each has 30 features)

budget = 1000
true_cost= np.array([0.8, 1, 0.6])

P = D.dot(th.T)
optimal = P.max(axis=1)
plt.subplot(2, 1, 1)
plt.title("Distribution of ideal arm choices")
fig = plt.hist(optimal, bins=range(0, n_a))

eps = 0.2
optimal_reward = []
choices = np.zeros(n, dtype=int)
rewards = np.zeros(n)
explore = np.zeros(n)
norms = np.zeros(n)
b = np.zeros_like(th)
A = np.zeros((n_a, k, k))
for a in range(0, n_a):
    A[a] = np.identity(k)
th_hat = np.zeros_like(th)  # our temporary feature vectors, our best current guesses
p = np.zeros(n_a)
alph = 0.2

# LINUCB, usign disjoint model
# This is all from Algorithm 1, p 664, "A contextual bandit appraoch..." Li, Langford
for i in range(0, n):
    x_i = D[i]  # the current context vector
    for a in range(0, n_a):
        A_inv = np.linalg.inv(A[a])  # we use it twice so cache it.
        th_hat[a] = A_inv.dot(b[a])  # Line 5
        ta = x_i.dot(A_inv).dot(x_i)  # how informative is this?
        a_upper_ci = alph * np.sqrt(ta)  # upper part of variance interval

        a_mean = th_hat[a].dot(x_i)  # current estimate of mean
        p[a] = (a_mean + a_upper_ci) / true_cost[a]
    norms[i] = np.linalg.norm(th_hat - th, 'fro')  # diagnostic, are we converging?
    # Let's hnot be biased with tiebraks, but add in some random noise
    p = p + (np.random.random(len(p)) * 0.000001)
    choices[i] = p.argmax()  # choose the highest, line 11

    # See what kind of result we get
    rewards[i] = P[i, choices[i]] / true_cost[choices[i]]  # using actual theta to figure out reward
    idx = np.argmax(P[i] / true_cost)
    # update the input vector
    A[choices[i]] += np.outer(x_i, x_i)
    b[choices[i]] +=  P[i, choices[i]] * x_i

    optimal_reward.append(P[i,idx] / true_cost[idx])


plt.figure(1, figsize=(10, 5))
plt.subplot(2, 1, 2)
plt.plot(norms)
plt.title("Frobeninus norm of estimated theta vs actual")

regret = (optimal_reward - rewards)
plt.subplot(2, 2, 2)
plt.plot(regret.cumsum())
plt.title("Cumulative regret")

plt.show()