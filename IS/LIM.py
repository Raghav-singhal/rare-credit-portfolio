import numpy as np
from scipy.stats import rv_discrete
def mutation_step((tn,Lt),T,params):
    #print "tn"+str(tn)
    N = params['numFirms']
    if(tn==T):
        return (tn,Lt)
    delta_t = Lambda_t(params,N,Lt)
    newtn = tn + delta_t
    newLt = Lt + 1.0
    if(newtn<=T):
        return (newtn,newLt)
    else:
        return (T,Lt)

def mutation(X_t,X_chi,T,params):
    M = X_t.shape[0]
    N = params['numFirms']
    for i in range(M):
        X_t[i],X_chi[i] = mutation_step((X_t[i],X_chi[i]),T,N,params)
    return X_t,X_chi

def Lambda_t(params,Lt):
    a = params['a']
    b = params['b']
    N = params['numFirms']
    if(Lt==0. or b==0.):
        return a
    else:
        scale_param = a*np.exp((b*Lt)/float(N))
        return np.random.exponential(scale=scale_param)

def selection(X_t,X_chi,alpha,T):
    M = X_t.shape[0]
    G = potential(X_t,M,alpha,T)
    norm_const = G.sum() / M
    probabilities = G / G.sum()
    sampled_indices = rv_discrete(
        values=(np.arange(M), probabilities)).rvs(size=M)
    X_t = X_t[sampled_indices]
    X_chi = X_chi[sampled_indices]
    return X_t,X_chi,norm_const


def potential(X_t,alpha,T):
    M = X_t.shape[0]
    probabilities = np.ones(M)
    for i in range(M):
        if(X_t[i]<T):
            probabilities[i] = np.exp(alpha)
    return probabilities

def estimator(X_chi,norm_consts,params):
    M = X_t.shape[0]
    N = params['numFirms']
    p = np.zeros(N+1)
    for i in range(M):
        p[int(X_chi[i])] = p[int(X_chi[i])] + np.exp(-1.0*params['alpha']*int(X_chi[i]))
    p = p/float(M)
    normalization = norm_consts.prod()
    p = p * normalization
    return p
def MCestimator(X_chi,norm_consts,params):
    M = X_t.shape[0]
    N = params['numFirms']
    p = np.zeros(N+1)
    for i in range(M):
        p[int(X_chi[i])] = p[int(X_chi[i])] + 1.0
    p = p/float(M)
    return p
