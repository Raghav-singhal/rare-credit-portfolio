import numpy as np
from scipy.stats import rv_discrete
import warnings
from tqdm import tqdm, tqdm_notebook
import parametricFns
import argparse
from multiprocessing import Pool,  freeze_support
import os
import pickle

def runMC(X0, params, barriers, tqdmParams):
    Xn = X0.copy()
    Xn = normalMC(Xn, params, parametricFns.A,
                          parametricFns.B, parametricFns.Cfn, tqdmParams)
    default_prob, defcounts = estimator(
        X0, Xn, barriers)
    return Xn, default_prob, defcounts
def estimator(X0, Xn,barrier):
    M = np.shape(X0)[0]  # Number of portfolios
    N = int((np.shape(X0)[1] - 1) / 2)  # Number of assets
    ndefaults = numDefaults(Xn[:, N + 1:], barrier)
    p = np.zeros(N + 1)
    defcounts = np.zeros(N + 1)
    for i in range(N + 1):
        ndef_i = (ndefaults == i)
        defcounts[i] = ndef_i.sum()
        p[i] = float(np.sum(ndef_i))/float(M)
    return p, defcounts
def numDefaults(minXn, barrier):
    return np.sum(minXn <= barrier, axis=1)
def MC_step(Xn, params, A, B, covariancefn):
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

def normalMC(Xn, params, A, B, covariancefn, tqdmParams=None):
    defaults = {'dt': 1e-3, 'T': 1.0, 'rho_sigma': -0.06, 'rho': 0.1,
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
    nTimesteps = int(params['T'] / params['dt'])
    if nTimesteps != params['T'] / params['dt']:
        warnings.warn(
            'dt does not evenly divide T so rounding num of steps to ' + str(nTimesteps))

    for _ in tqdml(range(nTimesteps), desc='mutation ', disable=tqdmParams['noverbose']):
        Xn = MC_step(Xn, params, A, B, covariancefn)
    return Xn

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='IPS sampling rare credit portfolio loss')
    parser.add_argument('--nportfolio', '-NP', default=20000, type=int,
                        help='Number of portfolios simulated (Default: %(default)s)')
    parser.add_argument('--nfirms', '-NF', default=125, type=int,
                        help='Number of firms in each portfolio (Default: %(default)s)')
    parser.add_argument('--maturity', '-T', default=1.0, type=float,
                        help='Maturity period of the portfolio in years (Default: %(default)s)')
    parser.add_argument('--nselection', '-NS', default=20, type=int,
                        help='Number of selection steps to be done (Default: %(default)s)')
    parser.add_argument('--mgranularity', '-dt', default=1e-3, type=float,
                        help='Granularity of mutation step in years. (Default: %(default)s)')
    parser.add_argument('--startprice', '-SP', default=90, type=float,
                        help='Initial price of assets (Default: %(default)s)')
    parser.add_argument('--sigma0', '-S0', default=0.4, type=float,
                        help='Idiosyncratic volatility. (Default: %(default)s)')
    parser.add_argument('--startvol', '-SV', default=0.4, type=float,
                        help='Initial stochastic volatility (Default: %(default)s)')
    parser.add_argument('--rate', '-R', default=0.06, type=float,
                        help='Risk free interest rate. (Default: %(default)s)')
    parser.add_argument('--alpha', nargs='*', default=[0.1], type=float,
                        help='alpha in potential function. Multiple values can be passed for averaging. (Default: %(default)s)')
    parser.add_argument('--barrier', default=36, type=float,
                        help='Barrier price for all assets (Default: %(default)s)')
    parser.add_argument('--deterministicvol', '-DV', action='store_true',
                        help='set to deterministic volatility only')
    parser.add_argument('--results', type=str,
                        help='Result directory to save to in results/ . Takes value based on parameters if unspecified')
    parser.add_argument('--jobs', type=int, default=4,
                        help='Worker jobs in pool (Default: %(default)s)')
    parser.add_argument('--noverbose', action='store_true',
                        help="Don't output any progress")
    parser.add_argument('--notebook', action='store_true',
                        help="running in notebook")
    args = parser.parse_args()

    if args.notebook:
        from tqdm import tqdm_notebook as tqdm

    initPrices = args.startprice * np.ones((args.nportfolio, args.nfirms))
    initVol = args.startvol * np.ones((args.nportfolio, 1))
    X0 = np.hstack((initVol, initPrices, initPrices))


    T = args.maturity
    barriers = args.barrier * np.ones(args.nfirms)
    #params = {'T': T, 'dt': args.mgranularity}
    params = {'T': T, 'dt': args.mgranularity,
              'sigma0': args.sigma0}

    Xn = []
    default_prob = []
    defcounts = []
    Xni,default_probi,defcountsi = runMC(X0,params,barriers,{'noverbose': args.noverbose, 'notebook': args.notebook})
    Xn.append(Xni)
    default_prob.append(default_probi)
    defcounts.append(defcountsi)
    maxDefAlphaInd = np.argmax(np.vstack(defcounts), axis=0)
    pkT = np.vstack(default_prob)[maxDefAlphaInd,
                                  np.arange(maxDefAlphaInd.shape[0])]
    #pkT = np.mean(np.vstack(default_prob), axis=0)

    results = {'args': args, 'params': params,'X0': X0, 'Xn': Xn,'default_prob': default_prob, 'pkT': pkT, 'defcounts': defcounts}

    resultDir = args.results or 'np' + str(args.nportfolio) + '_nf' + str(args.nfirms) + '_T' + str(T) + '_sp' + str(
        args.startprice) + '_sv' + str(args.startvol) + '_sigma' + str(args.sigma0) + '_DV' + str(args.deterministicvol)
    resultDir = 'results' + os.sep + resultDir
    if not os.path.exists(resultDir):
        os.makedirs(resultDir)

    with open(resultDir + os.sep + 'output.pkl', 'wb') as pfile:
        pickle.dump(results, pfile)
