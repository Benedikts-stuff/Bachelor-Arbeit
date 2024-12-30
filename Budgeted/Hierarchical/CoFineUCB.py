import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def learnU(W, K):
    A, sigma, B = np.linalg.svd(W, full_matrices=False)
    U_0 = A[:, :int(K)]
    W_0 = np.dot(U_0.T, W)
    X = np.dot(W_0, W_0.T)
    trace_X = np.trace(X)

    if trace_X == 0:
        raise ValueError("Trace of X is zero, cannot normalize Omega.")

    Omega = (K / trace_X) * X

    # Step 5: Return the projection matrix
    return np.dot(U_0, np.linalg.cholesky(Omega))



def CoFineUCB(D,N, lambda_fine, lambda_coarse, T, arms, theta):
    """
    CoFineUCB Algorithm Implementation

    Parameters:
    W: np.array
        User-item preference matrix (D x N).
    U: np.array
        Projection matrix (D x K), the coarse projection.
    lambda_fine: float
        Regularization parameter for the fine level.
    lambda_coarse: float
        Regularization parameter for the coarse level.
    T: int
        Number of rounds to run the algorithm.
    ct: callable
        Function representing context-specific uncertainty (fine level).
    c_tilde: callable
        Function representing context-specific uncertainty (coarse level).

    Returns:
    selected_actions: list
        List of selected actions (arms).
    rewards: list
        List of rewards received for each round.
    """
    W = np.random.uniform(0, 1, (D,N))  # D: number of items, N: number of users
    # im prizip habe ich hier den schritt des lernens der existierenden Nutzerprofile übergsprungen da ich direkt davon ausgehe das ich alle Profile kenne.
    # Eigentlich müsste der algorithmus nach und nach daten sammeln und mit einer metrik bestimmen welche Profile es gibt basierend auf seinen  beaobachtungen
    K = int(np.floor(D))  # Dimensionality of the latent space
    U = learnU(W,K)
    Xt = [[] for arm in range(arms)] # action history with weights of actions as elements

    Yt = [[] for arm in range(arms)]  # Yt: rewards observed

    opt_rewards = []
    observed_rewards = []

    for t in range(1, T + 1):
        ci = []
        # Step 3: Update Xt, X_tilde, and Yt
        user =  W[:, np.random.choice(N)] # ein user kommt auf die website

        if t <= arms:
            chosen_arm = t - 1
            arm = theta[chosen_arm]
            Xt[chosen_arm].append(user)

            observed_reward = np.dot(arm, user)
            observed_rewards.append(observed_reward)
            Yt[chosen_arm].append(observed_reward)
            opt_rewards.append(np.max(np.array([np.dot(w, user) for w in theta])))

            continue

        for arm_id in range(arms):
            X_tilde_t = U.T @ np.array(Xt[arm_id]).T  # Coarse level of Xt  # X_tilde_t is the projection onto the coarse space

            # Step 6: Compute M_tilde
            sq = X_tilde_t @ X_tilde_t.T
            M_tilde = lambda_coarse * np.identity(K) + sq

            # Step 7: Compute w_tilde using least squares on coarse level
            w_tilde = np.linalg.solve(M_tilde, X_tilde_t @ Yt[arm_id])

            # Step 8: Compute M_t for fine level
            sq_fine = np.array(Xt[arm_id]).T @ np.array(Xt[arm_id])
            M_fine = lambda_fine * np.identity(D) + sq_fine

            # Step 9: Compute w_t using least squares on fine level
            w_fine = np.linalg.solve(M_fine, np.array(Xt[arm_id]).T @ Yt[arm_id] + lambda_fine * U @ w_tilde)

            # Step 10: Compute μ_t(x) using the fine level weights
            def mu_t(x):
                return w_fine.T @ x

            def orth_vector(vec):
                x = np.random.randn(len(vec))
                x -= np.dot(x, vec) * vec / np.linalg.norm(vec)**2
                return x / np.linalg.norm(x)

            def compute_alpha_t():
                delta = 0.5
                alpha_t_v = np.sqrt(np.log(np.sqrt(np.linalg.det(M_fine)) * np.sqrt(np.linalg.det(lambda_fine * np.identity(D))))/delta)
                alpha_t_v_tilde = lambda_fine * np.sqrt(np.log(np.sqrt(np.linalg.det(M_tilde)) * np.sqrt(np.linalg.det(lambda_fine * np.identity(K))))/delta)
                alpha_t_b = np.sqrt(2) * lambda_fine * np.linalg.norm(orth_vector(w_fine))
                alpha_t_b_tilde = lambda_fine * lambda_coarse * np.linalg.norm(w_tilde)

                return alpha_t_b, alpha_t_v, alpha_t_b_tilde, alpha_t_v_tilde

            def compute_ct_and_ctilde(x):
                alpha_t_b, alpha_t_v, alpha_t_b_tilde, alpha_t_v_tilde = compute_alpha_t()
                M_inv_tilde = np.linalg.inv(M_tilde)
                M_inv = np.linalg.inv(M_fine)
                vector = M_inv_tilde.T @ (U.T @ np.linalg.inv(M_fine) @ x) @ M_inv_tilde
                c_t_tilde = alpha_t_v_tilde *  np.sqrt(vector.T @ M_inv_tilde @ vector) + alpha_t_b_tilde * np.linalg.norm(M_inv_tilde @ U.T @ M_inv @ x)
                c_t = alpha_t_b * np.sqrt(x.T @ M_inv @ x) + alpha_t_b * np.linalg.norm(M_inv @ x)

                return c_t, c_t_tilde

            mus = mu_t(user)
            c_t, c_t_tilde = compute_ct_and_ctilde(user)
            ci.append(mus + c_t + c_t_tilde)

        arm_idx = np.argmax(np.array(ci))
        arm = theta[arm_idx]
        Xt[arm_idx].append(user)

        observed_reward = np.dot(arm, user)
        observed_rewards.append(observed_reward)
        Yt[arm_idx].append(observed_reward)
        opt_arm_idx = np.argmax(np.array([np.dot(w, user) for w in theta]))
        opt_rewards.append(np.dot(theta[opt_arm_idx], user))
        print("optimal arm: ", opt_arm_idx, "played arm: ", arm_idx)

    return observed_rewards, opt_rewards


# Beispielaufruf
W = np.random.uniform(low=0.0, high=1.0, size=(100, 10))
D = 100
N = 200
lambda_fine = 1
lambda_coarse = 1
T = 10000
arms = 5
theta = np.random.uniform(0, 1, (5,100))

rewards, opt_rewards = CoFineUCB(D,N, lambda_fine, lambda_coarse, T, arms, theta)
plt.plot(np.cumsum(np.array(opt_rewards) - rewards), label='regret')
plt.title("Cumulative regret")
plt.legend()
plt.show()