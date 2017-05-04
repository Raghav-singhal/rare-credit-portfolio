import numpy as np


def A(portfolio, params):
    sigma, S = portfolio[:, :1], portfolio[:, 1:]
    return np.hstack((params['kappa'] * (params['sigmaHat'] - sigma), params['r'] * S))


def B(portfolio, params):
    sigma, S = portfolio[:, :1], portfolio[:, 1:]
    return np.hstack((params['gamma'] * np.sqrt(sigma), params['sigma0'] * sigma * S))


def Cfn(N, params):
    N += 1
    C = params['rho'] * params['dt'] * np.ones((N, N))
    np.fill_diagonal(C, params['dt'])
    C[0, 1:] = params['rho_sigma'] * params['dt']
    C[1:, 0] = params['rho_sigma'] * params['dt']
    return C
