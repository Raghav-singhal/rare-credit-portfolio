import numpy as np
from scipy.stats import rv_discrete
import warnings
from tqdm import tqdm, tqdm_notebook

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
# alpha - chosen alpha paramter
# potentialfn - function which returns the potential
# Returns:
# Xnew - matrix of size numPortfolios x (2N+1). Selected new values
# norm_const - scalar normalization constant at this selection stage.
# Needs to be accumulated.


def selection(Xp, Wp, alpha, potentialfn=potential):
    numPortfolios = Xp.shape[0]
    n = int((Xp.shape[1] - 1) / 2)
    G = potentialfn(Xp, Wp, alpha)
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


def mutation(Xn, params, A, B, covariancefn, tqdmParams=None):
    defaults = {'dt': 1e-3, 'Dt': 0.05, 'rho_sigma': -0.06, 'rho': 0.1,
                'kappa': 3.5, 'sigmaHat': 0.4, 'r': 0.06, 'gamma': 0.7,
                'sigma0': 0.4}
    for defkey in defaults.keys():
        params.setdefault(defkey, defaults[defkey])
    tqdmDefaults = {'nFn': 0, 'noverbose': False, 'notebook': False}
    if tqdmParams is None:
        tqdmParams = {}
    for defkey in tqdmDefaults.keys():
        tqdmParams.setdefault(defkey, tqdmDefaults[defkey])
    tqdml = tqdm
    if tqdmParams['notebook']:
        tqdml = tqdm_notebook
    nTimesteps = int(params['Dt'] / params['dt'])
    if nTimesteps != params['Dt'] / params['dt']:
        warnings.warn(
            'dt does not evenly divide Dt so rounding num of steps to ' + str(nTimesteps))
    Wn = Xn.copy()
    for _ in tqdml(range(nTimesteps), desc='mutation ' + str(tqdmParams['nFn']), position=2 * tqdmParams['nFn'] + 1, disable=tqdmParams['noverbose']):
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
def estimator(X0, Xn,  Wn, barrier, alpha, norm_consts):
    M = np.shape(X0)[0]  # Number of portfolios
    N = int((np.shape(X0)[1] - 1) / 2)  # Number of assets
    G = potential(X0, Wn, alpha)
    ndefaults = numDefaults(Xn[:, N + 1:], barrier)
    p = np.zeros(N + 1)
    normalization = norm_consts.prod()
    defcounts = np.zeros(N + 1)
    for i in range(N + 1):
        ndef_i = (ndefaults == i)
        defcounts[i] = ndef_i.sum()
        p[i] = np.sum(G * ndef_i) * normalization / M
    return p, defcounts

# X0 - numPortfolios x N matrix of initial state.
# barrier - numpy array of size N for barrier prices of assets


def MCestimator(Xn, barrier):
    M = np.shape(Xn)[0]  # Number of portfolios
    N = int((np.shape(Xn)[1] - 1) / 2)  # Number of assets
    ndefaults = numDefaults(Xn[:, N + 1:], barrier)
    defcounts = np.zeros(N + 1)
    for i in range(N + 1):
        ndef_i = (ndefaults == i)
        defcounts[i] = ndef_i.sum()
    p = defcounts / defcounts.sum()
    return p, defcounts
