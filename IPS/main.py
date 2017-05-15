import argparse
import numpy as np
import IPS
import parametricFns
import LIM
try:
    import cPickle as pickle
except:
    import pickle
from tqdm import tqdm
from multiprocessing import Pool,  freeze_support
import os


def runIPS(X0, params, n, alpha, barriers, Afn, Bfn, Cfn, tqdmParams):
    print("Estimating using Merton IPS")
    Xn = X0.copy()
    Wn = X0.copy()
    norm_consts = []
    for i in tqdm(range(n), desc='Mertons Model selection ' + str(tqdmParams['nFn']), position=2 * tqdmParams['nFn'],  disable=tqdmParams['noverbose']):
        Xn, nc = IPS.selection(Xn, Wn, alpha)
        Xn, Wn = IPS.mutation(Xn, params, Afn,
                              Bfn, Cfn, tqdmParams)
        norm_consts.append(nc)
    norm_consts = np.array(norm_consts)
    default_prob, defcounts = IPS.estimator(
        X0, Xn, Wn, barriers, alpha, norm_consts)
    return Xn, Wn, norm_consts, default_prob, defcounts


def runMC(X0, params, n, alpha, barriers, Afn, Bfn, Cfn, tqdmParams):
    print("Estimating using Merton Simple Monte Carlo")
    Xn = X0.copy()
    Xn, _ = IPS.mutation(Xn, params, Afn, Bfn, Cfn, tqdmParams)
    default_prob, defcounts = IPS.MCestimator(Xn, barriers)
    return Xn, None, None, default_prob, defcounts


def runLIMMC(X_0, params, n, alpha, barriers, Afn, Bfn, Cfn, tqdmParams):
    print("Estimating using LIM Simple Monte Carlo")
    Xp = X_0.copy()
    params['alpha'] = alpha
    N = params['numFirms']
    T = params['T']
    for n in tqdm(range(N)):
        Wp = Xp.copy()
        Xp = LIM.mutation(Xp, params)
    default_prob, def_counts = LIM.MCestimator(Xp, norm_consts, params)
    return Xp, Wp, None, default_prob, defcounts


def runLIM(X_0, params, n, alpha, barriers, Afn, Bfn, Cfn, tqdmParams):
    Xp = X_0.copy()
    params['alpha'] = alpha
    N = params['numFirms']
    T = params['T']
    norm_consts = []
    for n in tqdm(range(N), desc='Local Initialization Model selection ' + str(tqdmParams['nFn']), position=2 * tqdmParams['nFn']):
        Wp = Xp.copy()
        Xp, norm_const = LIM.selection(Xp, params)
        Xp = LIM.mutation(Xp, params)
        norm_consts.append(norm_const)
    norm_consts = np.array(norm_consts)
    default_prob, defcounts = LIM.estimator(Xp, norm_consts, params)
    return Xp, Wp, norm_consts, default_prob, defcounts


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
    parser.add_argument('--mconly', '-MC', action='store_true',
                        help='Use MC sampling only otherwise normally runs IPS')
    parser.add_argument('--lim', '-LIM', action='store_true',
                        help='Set the model to local intensity model')
    parser.add_argument('--results', type=str, default='results',
                        help='Result directory to save to in results. (Default: %(default)s)')
    parser.add_argument('--jobs', type=int, default=4,
                        help='Worker jobs in pool (Default: %(default)s)')
    parser.add_argument('--noverbose', action='store_true',
                        help="Don't output any progress")
    parser.add_argument('--notebook', action='store_true',
                        help="running in notebook")
    parser.add_argument('--a', '-a', type=float, default=0.01,
                        help="Parameter of lambda function in LIM (Default: %(default)s)")
    parser.add_argument('--b', '-b', type=float, default=13,
                        help="Parameter of lambda function in LIM (Default: %(default)s)")

    args = parser.parse_args()

    if args.notebook:
        from tqdm import tqdm_notebook as tqdm
    T = args.maturity
    alpha = args.alpha

    if args.deterministicvol:
        args.startvol = 1

    initPrices = args.startprice * np.ones((args.nportfolio, args.nfirms))
    initVol = args.startvol * np.ones((args.nportfolio, 1))
    X0 = np.hstack((initVol, initPrices, initPrices))
    n = args.nselection
    Dt = T / n
    barriers = args.barrier * np.ones(args.nfirms)

    print(args.mconly)

    if args.lim:
        runFn = runLIM
        X0 = np.zeros((args.nportfolio, 2))
        params = {'a': args.a, 'b': args.b, 'numFirms': args.nfirms, 'T': T}
        if args.mconly:
            runFn = runLIMMC
            alpha = [1]
    else:
        runFn = runIPS
        params = {'Dt': Dt, 'dt': args.mgranularity,
                  'sigma0': args.sigma0, 'r': args.rate}
        if args.mconly:
            runFn = runMC
            alpha = [1]
            Dt = T
            n = 1

    Afn = parametricFns.A
    Bfn = parametricFns.B
    Cfn = parametricFns.Cfn
    if args.deterministicvol and not args.lim:
        params['kappa'] = 0
        params['gamma'] = 0
        Cfn = parametricFns.Cfn_no_stoch_vol

    Xn = []
    Wn = []
    norm_consts = []
    default_prob = []
    defcounts = []
    # runFn(X0,params,n,None,barriers,Afn,Bfn,Cfn,None)
    with Pool(processes=args.jobs) as pool:
        multipleresults = [pool.apply_async(
            runFn, (X0, params, n, alpha[i], barriers, Afn, Bfn, Cfn,
                    {'nFn': i, 'noverbose': args.noverbose,
                     'notebook': args.notebook})) for i in range(len(alpha))]
        for i in range(len(alpha)):
            Xni, Wni, norm_constsi, default_probi, defcountsi = multipleresults[i].get(
            )
            Xn.append(Xni)
            Wn.append(Wni)
            norm_consts.append(norm_constsi)
            default_prob.append(default_probi)
            defcounts.append(defcountsi)
    maxDefAlphaInd = np.argmax(np.vstack(defcounts), axis=0)
    pkT = np.vstack(default_prob)[maxDefAlphaInd,
                                  np.arange(maxDefAlphaInd.shape[0])]
    pkT = np.mean(np.vstack(default_prob), axis=0)

    results = {'args': args, 'params': params, 'alpha': alpha, 'X0': X0, 'Xn': Xn,
               'Wn': Wn, 'norm_consts': norm_consts, 'default_prob': default_prob,
               'pkT': pkT, 'defcounts': defcounts, 'barriers': barriers}

    if args.lim:
        modelName = "LIM"
        resultDir = modelName + '_np' + str(args.nportfolio) + '_nf' + str(args.nfirms) + '_T' \
            + str(T) + '_MC' + str(args.mconly) + '_alpha' \
            + str(alpha[0]) + '_' + str(alpha[-1]) + '_' + \
            str(len(alpha)) + '_b' + str(args.b)

    else:
        modelName = "Merton"
        resultDir = modelName + '_np' + str(args.nportfolio) + '_nf' + str(args.nfirms) + '_T' \
            + str(T) + '_ns' + str(n) + '_sp' + str(args.startprice) + '_sv' \
            + str(args.startvol) + '_sigma' + str(args.sigma0) + '_DV' \
            + str(args.deterministicvol) + '_MC' + str(args.mconly) + '_alpha' \
            + str(alpha[0]) + '_' + str(alpha[-1]) + '_' + str(len(alpha))
    resultDir = args.results + os.sep + resultDir
    os.makedirs(resultDir, exist_ok=True)

    with open(resultDir + os.sep + 'output.pkl', 'wb') as pfile:
        pickle.dump(results, pfile)
