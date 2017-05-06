import argparse
import numpy as np
import IPS
import parametricFns
import pickle
from tqdm import tqdm
from multiprocessing import Pool,  freeze_support
import os


def runIPS(X0, params, n, alpha, barriers, tqdmParams):
    Xn = X0.copy()
    Wn = X0.copy()
    norm_consts = []
    for i in tqdm(range(n), desc='selection ' + str(tqdmParams['nFn']), position=2 * tqdmParams['nFn'],  disable=tqdmParams['noverbose']):
        Xn, nc = IPS.selection(Xn, Wn, alpha)
        Xn, Wn = IPS.mutation(Xn, params, parametricFns.A,
                              parametricFns.B, parametricFns.Cfn, tqdmParams)
        norm_consts.append(nc)
    norm_consts = np.array(norm_consts)
    default_prob, defcounts = IPS.estimator(
        X0, Xn, Wn, barriers, alpha, norm_consts)
    return Xn, Wn, norm_consts, default_prob, defcounts


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
    parser.add_argument('--startvol', '-SV', default=0.4, type=float,
                        help='Initial stochastic volatility (Default: %(default)s)')
    parser.add_argument('--alpha', nargs='*', default=[0.1], type=float,
                        help='alpha in potential function. Multiple values can be passed for averaging. (Default: %(default)s)')
    parser.add_argument('--barrier', default=36, type=float,
                        help='Barrier price for all assets (Default: %(default)s)')
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

    n = args.nselection
    T = args.maturity
    Dt = T / n
    alpha = args.alpha
    barriers = args.barrier * np.ones(args.nfirms)

    params = {'Dt': Dt, 'dt': args.mgranularity}

    Xn = []
    Wn = []
    norm_consts = []
    default_prob = []
    defcounts = []
    with Pool(processes=args.jobs) as pool:
        multipleresults = [pool.apply_async(
            runIPS, (X0, params, n, alpha[i], barriers, {'nFn': i, 'noverbose': args.noverbose, 'notebook': args.notebook})) for i in range(len(alpha))]
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
    #pkT = np.mean(np.vstack(default_prob), axis=0)

    results = {'args': args, 'params': params, 'alpha': alpha, 'X0': X0, 'Xn': Xn,
               'Wn': Wn, 'norm_consts': norm_consts, 'default_prob': default_prob, 'pkT': pkT, 'defcounts': defcounts}

    resultDir = args.results or 'np' + str(args.nportfolio) + '_nf' + str(args.nfirms) + '_T' + str(T) + '_ns' + str(n) + '_sp' + str(
        args.startprice) + '_sv' + str(args.startvol) + '_alpha' + str(alpha[0]) + '_' + str(alpha[-1]) + '_' + str(len(alpha))
    resultDir = 'results' + os.sep + resultDir
    os.makedirs(resultDir, exist_ok=True)

    with open(resultDir + os.sep + 'output.pkl', 'wb') as pfile:
        pickle.dump(results, pfile)
