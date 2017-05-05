import numpy as np
from scipy.stats import rv_discrete
import warnings

# Xp, Wp - matrices of size numPortfolios x (2N+1). Current state and history.
# Returns:
# G - numPortfolios size array of the multiplicative potentials
# G = exp[-\alpha (V(X_p) - V(X_{p-1}))]
# where V(X_p) = \sum_{i = 1}^{N} \log (\min_{0 \leq m \leq p} S_i(m \Delta t))


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


# Xn- matrix of size numPortfolios x (2N+1). Current state.
# params - dict containing paramters used in model
# A, B, covariancefn - parameters which are functions used as follow:
# Xn+1 = Xn + a(Xn)dt + b(Xn)dW
# covariancefn gives the covariance matrix of Weiner process.
# Returns Xn+1
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

# Performs mutation_step multiple times so that multiple dt timesteps are
# taken to give a time interval of Dt.
# Returns Xn+1, Wn+1 - previous and current mutated state.


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


# minXn - numPortfolios x N matrix. Has minimum seen prices of N assets in all portfolios
# barrier - Numpy array of size N indicating barrier price for each asset
# Returns numpy array of size M indicating number of defaults for each
# portfolio
def numDefaults(minXn, barrier):
    return np.sum(minXn <= barrier, axis=1)


# X0 - numPortfolios x N matrix of initial state.
# Wn - numPortfolios x N matrix of state before last mutation
# barrier - numpy array of size N for barrier prices of assets
# alpha - Potential function parameter
# norm_consts - numpy array of size n with normalization constants
# calculated at each selection stage.
def estimator(X0, Wn, barrier, alpha, norm_consts):
    M = np.shape(X0)[0]  # Number of portfolios
    N = int((np.shape(X0)[1] - 1) / 2)  # Number of assets
    G = potential(X0, Wn, alpha)
    ndefaults = numDefaults(X0[:, N + 1:], barrier)
    p = np.zeros(N + 1)
    normalization = norm_consts.prod()
    for i in range(N + 1):
        p[i] = np.sum(G * (ndefaults == i)) * normalization / M
    return p
