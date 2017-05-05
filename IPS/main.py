import argparse
import numpy as np
import IPS
import parametricFns
import pickle
from tqdm import tqdm

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
parser.add_argument('--startprice', '-SP', default=90, type=float,
                    help='Initial price of assets (Default: %(default)s)')
parser.add_argument('--startvol', '-SV', default=0.4, type=float,
                    help='Initial stochastic volatility (Default: %(default)s)')
parser.add_argument('--alpha', default=0.1, type=float,
                    help='alpha in potential function (Default: %(default)s)')
parser.add_argument('--barrier', default=36, type=float,
                    help='Barrier price for all assets (Default: %(default)s)')
args = parser.parse_args()


initPrices = args.startprice * np.ones((args.nportfolio, args.nfirms))
initVol = args.startvol * np.ones((args.nportfolio, 1))
X0 = np.hstack((initVol, initPrices, initPrices))
Xn = X0.copy()
Wn = X0.copy()

n = args.nselection
T = args.maturity
Dt = T / n
alpha = args.alpha
barriers = args.barrier * np.ones(args.nfirms)

params = {'Dt': Dt}

norm_consts = []

#Xn_bs = []
#Xn_bm = []
#Xn_am = []
#Wn_am = []

for i in tqdm(range(n)):
    # Xn_bs.append(Xn.copy())
    Xn, nc = IPS.selection(Xn, Wn, alpha)
    # Xn_bm.append(Xn.copy())
    Xn, Wn = IPS.mutation(Xn, params, parametricFns.A,
                          parametricFns.B, parametricFns.Cfn)
    # Xn_am.append(Xn.copy())
    # Wn_am.append(Wn.copy())
    norm_consts.append(nc)
norm_consts = np.array(norm_consts)
default_prob = IPS.estimator(X0, Xn, Wn, barriers, alpha, norm_consts)
