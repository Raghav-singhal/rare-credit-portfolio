import numpy as np
from scipy.stats import rv_discrete
import warnings

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


def mutation_step(Xn, params, A, B, covariancefn):
    M = np.shape(Xn)[0]  # Number of portfolios
    N = int((np.shape(Xn)[1] - 1) / 2)  # Number of assets
    C = covariancefn(N, params)
    X = Xn[:, :N + 1]
    minX = Xn[:, N + 1:]
    a1 = A(X, params)
    b1 = B(X, params)
    dW = np.sqrt(params['dt']) * \
        np.random.multivariate_normal(np.zeros(N + 1), C, M)
    X += a1 * params['dt'] + b1 * dW
    np.copyto(minX, np.minimum(X[:, 1:], minX))
    return Xn


def mutation(Xn, params, A, B, covariancefn):
    defaults = {'dt': 1e-4, 'Dt': 0.05, 'rho_sigma': -0.06, 'rho': 0.1,
                'kappa': 3.5, 'sigmaHat': 0.4, 'r': 0.06, 'gamma': 0.7,
                'sigma0': 0.4}
    for defkey in defaults.keys():
        params.setdefault(defkey, defaults[defkey])
    nTimesteps = int(params['Dt'] / params['dt'])
    if nTimesteps != params['Dt'] / params['dt']:
        warnings.warn(
            'dt does not evenly divide Dt so rounding num of steps to ' + str(nTimesteps))
    Wn = Xn.copy()
    for _ in range(nTimesteps):
        Xn = mutation_step(Xn, params, A, B, covariancefn)
    return Xn, Wn
