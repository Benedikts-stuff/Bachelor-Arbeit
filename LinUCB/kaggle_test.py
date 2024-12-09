
import numpy as np
from matplotlib import pyplot as plt


np.random.seed(42)
#d = 3 #features.shape[1]
n_a = 400 #df['campaign_id'].nunique()  # Anzahl der eindeutigen Kampagnen
k= 5 # number of features
n =5000
D = np.random.random((n, k)) - 0.5 # our data, or these are the contexts, there are n contexts, each has k features

th = np.random.random((n_a, k)) - 0.5
eps = 0.2
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
P = D.dot(th.T)
means = np.mean(P, axis=0)
# LINUCB, usign disjoint model
# This is all from Algorithm 1, p 664, "A contextual bandit appraoch..." Li, Langford
for i in range(0, n):
    x_i = D[i]  # the current context vector
    for a in range(0, n_a):
        A_inv = np.linalg.inv(A[a])  # we use it twice so cache it.
        th_hat[a] = A_inv.dot(b[a])  # Line 5
        ta = x_i.dot(A_inv).dot(x_i)  # how informative is this?
        a_upper_ci = (1 + np.sqrt(np.log(2 * (i+ 1)) / 2)) * np.sqrt(ta)  # upper part of variance interval

        a_mean = th_hat[a].dot(x_i)  # current estimate of mean
        p[a] = a_mean + a_upper_ci
    norms[i] = np.linalg.norm(th_hat - th, 'fro')  # diagnostic, are we converging?
    # Let's hnot be biased with tiebraks, but add in some random noise
    p = p + (np.random.random(len(p)) *0.000001)
    choices[i] = p.argmax()  # choose the highest, line 11

    # See what kind of result we get
    rewards[i] = th[choices[i]].dot(x_i)  # using actual theta to figure out reward

    # update the input vector
    A[choices[i]] += np.outer(x_i, x_i)
    b[choices[i]] += rewards[i] * x_i



plt.figure(1, figsize=(10, 5))
plt.subplot(121)
plt.plot(norms)
plt.title("Frobeninus norm of estimated theta vs actual")
plt.show()


plt.subplot(122)
plt.plot(rewards.cumsum(), label= 'linear_model')
plt.title("Cumulative reward")
plt.legend()
plt.show()