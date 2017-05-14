import numpy as np
from scipy.stats import rv_discrete


"""
Takes the next jump in the mutation process
where jump time is sampled from exponential distribution
and we progress to next t only if the next t is less than
maturity period and next step is calculated accordingly.
"""
def mutation_step(X,T,params):
    tn = X[0]
    Lt = X[1]
    N = params['numFirms']
    if(tn==T):
        return (tn,Lt)
    delta_t = Lambda_t(params,Lt)
    newtn = tn + delta_t
    newLt = Lt + 1.0
    if(newtn<=T):
        return (newtn,newLt)
    else:
        return (T,Lt)

"""
Performs mutation steps for M portfolios
"""

def mutation(X_t,X_chi,params):
    T = params['T']
    M = X_t.shape[0]
    N = params['numFirms']
    for i in range(M):
        X_t[i],X_chi[i] = mutation_step((X_t[i],X_chi[i]),T,params)
    return X_t,X_chi

"""
Sample function for the jump time
"""

def Lambda_t(params,Lt):
    a = params['a']
    b = params['b']
    N = params['numFirms']
    scale_param = 1.0/(a*np.exp((b*Lt)/float(N)))
    return np.random.exponential(scale=scale_param)

"""
Selection process similar to Merton's model
"""

def selection(X_t,X_chi,params):
    alpha = params['alpha']
    T = params['T']
    M = X_t.shape[0]
    G = potential(X_t,alpha,T)
    norm_const = G.sum() / M
    probabilities = G / G.sum()
    sampled_indices = rv_discrete(
        values=(np.arange(M), probabilities)).rvs(size=M)
    X_t = X_t[sampled_indices]
    X_chi = X_chi[sampled_indices]
    return X_t,X_chi,norm_const
"""
Potential function for the Interacting particle system:
if t<T then it's e^alpha else it's 1.0
"""

def potential(X_t,alpha,T):
    M = X_t.shape[0]
    probabilities = np.ones(M)
    for i in range(M):
        if(X_t[i]<T):
            probabilities[i] = np.exp(alpha)
    return probabilities

"""
Estimating probabilities counting the expectation of defaults
at 125th iteration.Here the counts are scaled with potential
and then normalized with expectation of normalization constants.
"""
def estimator(X_chi,norm_consts,params):
    M = X_chi.shape[0]
    N = params['numFirms']
    p = np.zeros(N+1)
    for i in range(M):
        p[int(X_chi[i])] = p[int(X_chi[i])] + np.exp(-1.0*params['alpha']*int(X_chi[i]))
    defcounts = p
    normalization = norm_consts[1:].prod()
    default_prob = (p/float(M))*normalization
    return default_prob, defcounts
"""
Estimating probabilities counting the expectation of defaults
at 125th iteration.Unscaled with potential and no norm consts
because of absence of potentials.
"""

def MCestimator(X_chi,norm_consts,params):
    M = X_chi.shape[0]
    N = params['numFirms']
    p = np.zeros(N+1)
    for i in range(M):
        p[int(X_chi[i])] = p[int(X_chi[i])] + int(X_chi[i])
    defcounts = p
    default_prob = p/float(M)
    return default_prob, defcounts
