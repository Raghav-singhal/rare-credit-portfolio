import numpy as np
from scipy.stats import rv_discrete

# Xp, Wp - matrices of size numPortfolios x (2N+1). Current state and history.
# Returns:
# G - numPortfolios size array of the multiplicative potentials


def potential(Xp, Wp, alpha):
    numPortfolios = Xp.shape[0]
    n = int((Xp.shape[1] - 1) / 2)
    min_Xp = Xp[:, -n:]
    min_Wp = Wp[:, -n:]
    VXp = np.sum(np.log(min_Xp), axis=1)
    VWp = np.sum(np.log(min_Wp), axis=1)
    G = np.exp(-alpha * (VXp - VWp))
    return G

# Xp, Wp - matrices of size numPortfolios x (2N+1). Current state and history.
# Returns:
# Xnew - matrix of size numPortfolios x (2N+1). Selected new values
# norm_const - scalar normalization constant at this selection stage.
# Needs to be accumulated.


def selection(Xp, Wp, alpha):
    numPortfolios = Xp.shape[0]
    n = int((Xp.shape[1] - 1) / 2)
    G = potential(Xp, Wp, alpha)
    norm_const = G.sum() / numPortfolios
    probabilities = G / G.sum()
    sampled_indices = rv_discrete(
        values=(np.arange(numPortfolios), probabilities)).rvs(size=numPortfolios)
    Xnew = Xp[sampled_indices, :]
    return Xnew, norm_const
