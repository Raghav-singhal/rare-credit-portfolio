import numpy as np
from scipy.stats import rv_discrete

# Beta functions below return 1/lambda value for use in exponential dist


def beta_t_1(Lt, params):
    a = params['a']
    b = params['b']
    N = params['numFirms']
    beta_t = 1 / (a * np.exp(b * Lt / N))
    return beta_t


def beta_t_2(Lt, params):
    a = params['a']
    b = params['b']
    N = params['numFirms']
    beta_t = 1 / (1 - Lt / N)
    return beta_t

# Performs mutation steps for M portfolios


def mutation(Xp, params, betafn=beta_t_2):
    T = params['T']
    M = Xp.shape[0]
    N = params['numFirms']
    X_t = Xp[:, 0]
    X_xi = Xp[:, 1]
    beta_t = betafn(X_xi, params)
    delta_t = beta_t * np.random.exponential(size=M)
    X_t += delta_t
    X_xi[X_t <= T] += 1
    X_t[X_t > T] = T
    return Xp

# Selection process similar to Merton's model


def selection(Xp, params):
    alpha = params['alpha']
    T = params['T']
    M = Xp.shape[0]
    X_t = Xp[:, 0]
    G = potential(X_t, alpha, T)
    norm_const = G.sum() / M
    probabilities = G / G.sum()
    sampled_indices = rv_discrete(
        values=(np.arange(M), probabilities)).rvs(size=M)
    Xnew = Xp[sampled_indices, :]
    return Xnew, norm_const

# Potential function for the Interacting particle system:
# if t<T then it's e^alpha else it's 1.0


def potential(X_t, alpha, T):
    M = X_t.shape[0]
    probabilities = np.ones(M)
    probabilities[X_t < T] = np.exp(alpha)
    return probabilities

# Estimating probabilities counting the expectation of defaults
# at 125th iteration.Here the counts are scaled with potential
# and then normalized with expectation of normalization constants.


def estimator(Xp, norm_consts, params):
    M = Xp.shape[0]
    N = params['numFirms']
    p = np.zeros(N + 1)
    defcounts = np.zeros(N + 1)
    X_xi = Xp[:, 1]
    np.add.at(p, X_xi.astype(np.int), np.exp(-params['alpha'] * X_xi))
    np.add.at(defcounts, X_xi.astype(np.int), 1)
    normalization = norm_consts[1:].prod()
    default_prob = p / M * normalization
    return default_prob, defcounts

# Estimating probabilities counting the expectation of defaults
# at 125th iteration.Unscaled with potential and no norm consts
# because of absence of potentials.


def MCestimator(Xp, norm_consts, params):
    M = Xp.shape[0]
    N = params['numFirms']
    defcounts = np.zeros(N + 1)
    X_xi = Xp[:, 1]
    np.add.at(defcounts, X_xi.astype(np.int), 1)
    default_prob = defcounts / defcounts.sum()
    return default_prob, defcounts
