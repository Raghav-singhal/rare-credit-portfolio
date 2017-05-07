import numpy as np


# a(Xn) = [kappa*(sigmaHat - sigma(n)) ,r*S1,..,r*SN]
def A(portfolio, params):
    sigma, S = portfolio[:, :1], portfolio[:, 1:]
    return np.hstack((params['kappa'] * (params['sigmaHat'] - sigma), params['r'] * S))


# b(Xn) =
# [gamma*np.sqrt(sigma(n)),simga0*sigma(n)*S1,.....,simga0*sigma(n)*SN]
def B(portfolio, params):
    sigma, S = portfolio[:, :1], portfolio[:, 1:]
    return np.hstack((params['gamma'] * np.sqrt(sigma), params['sigma0'] * sigma * S))


def Cfn(N, params):
    N += 1
    C = params['rho'] * np.ones((N, N))
    np.fill_diagonal(C, 1)
    C[0, 1:] = params['rho_sigma']
    C[1:, 0] = params['rho_sigma']
    return C


def Cfn_no_stoch_vol(N, params):
    N += 1
    C = params['rho'] * np.ones((N, N))
    np.fill_diagonal(C, 1)
    C[0, 1:] = 0
    C[1:, 0] = 0
    return C
