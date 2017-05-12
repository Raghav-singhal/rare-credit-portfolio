import numpy as np
import argparse
import LIM
import os
import pickle
"""
This main is for running Local Intensity Model
"""
def runLIM(numPortfolios,numFirms,T,defaults):
    M = numPortfolios
    N = numFirms
    params = defaults
    X_t,X_chi,norm_consts = LIM.initialize(M,N)
    for n in range(N):
        W_t = X_t.copy()
        W_chi = X_chi.copy()
        Xn_t,Xn_chi,norm_const = LIM.selection(W_t,W_chi,M,params['alpha'],T)
        print Xn_t
        Xn_t,Xn_chi = LIM.mutation(Xn_t,Xn_chi,M,N,T,params)
        print Xn_t
        X_t = Xn_t.copy()
        X_chi = Xn_chi.copy()
        #print X_t-W_t
        norm_consts[n+1] = norm_const

    return X_t,X_chi,norm_consts

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='IPS sampling rare credit portfolio loss')
    parser.add_argument('--nportfolio', '-NP', default=20000, type=int,
                        help='Number of portfolios simulated (Default: %(default)s)')
    parser.add_argument('--nfirms', '-NF', default=125, type=int,
                        help='Number of firms in each portfolio (Default: %(default)s)')
    parser.add_argument('--maturity', '-T', default=1.0, type=float,
                        help='Maturity period of the portfolio in years (Default: %(default)s)')
    parser.add_argument('--alpha', nargs='*', default=[0.1], type=float,
                        help='alpha in potential function. Multiple values can be passed for averaging. (Default: %(default)s)')
    parser.add_argument('--results', type=str,
                        help='Result directory to save to in results/ . Takes value based on parameters if unspecified')
    parser.add_argument('--jobs', type=int, default=4,
                        help='Worker jobs in pool (Default: %(default)s)')
    parser.add_argument('--noverbose', action='store_true',
                        help="Don't output any progress")
    parser.add_argument('--notebook', action='store_true',
                        help="running in notebook")
    parser.add_argument('--model', '-md', default='LIM',type=str,
                         help="Name of the model to estimate")
    args = parser.parse_args()

    args.nportfolio
    if args.notebook:
        from tqdm import tqdm_notebook as tqdm
    defaults = {'alpha':0.4,'a':0.01,'b':13}
    X_t,X_chi,norm_consts = runLIM(args.nportfolio,args.nfirms,args.maturity,defaults)
    pkT = LIM.estimator(args.nportfolio,args.nfirms,X_chi,norm_consts,defaults)


    results = {'modelName':'LIM','args': args, 'params': defaults, 'X_t':X_t, 'X_chi':X_chi,
               'norm_consts': norm_consts, 'pkT': pkT}
    resultDir = args.results or 'np' + str(args.nportfolio) + '_nf' + str(args.nfirms) + '_T' + str(args.maturity) + '_alpha' + str(defaults['alpha'])
    resultDir = 'results' + os.sep + resultDir
    if not os.path.exists(resultDir):
        os.makedirs(resultDir)

    with open(resultDir + os.sep + 'output.pkl', 'wb') as pfile:
        pickle.dump(results, pfile)
