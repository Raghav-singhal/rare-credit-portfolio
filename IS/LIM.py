import numpy as np
from scipy.stats import rv_discrete
def mutation_step((tn,Lt),T,N,params):
    #print "tn"+str(tn)
    if(tn==T):
        return (tn,Lt)
    delta_t = Lambda_t(params,N,Lt)
    newtn = tn + delta_t
    newLt = Lt + 1.0
    if(newtn<=T):
        return (newtn,newLt)
    else:
        return (T,Lt)

def mutation(X_t,X_chi,M,N,T,params):
    for i in range(M):
        X_t[i],X_chi[i] = mutation_step((X_t[i],X_chi[i]),T,N,params)
    return X_t,X_chi

def Lambda_t(params,N,Lt):
    a = params['a']
    b = params['b']
    #print "lambda: "+str(a*np.exp(b*Lt/float(N)))
    if(Lt==0. or b==0.):
        return a
    else:
        scale_param = float(N)/(Lt*b)
        return a*scale_param*np.random.exponential(scale=scale_param)

def selection(X_t,X_chi,M,alpha,T):
    G = potential(X_t,M,alpha,T)
    norm_const = G.sum() / M
    probabilities = G / G.sum()
    sampled_indices = rv_discrete(
        values=(np.arange(M), probabilities)).rvs(size=M)
    X_t = X_t[sampled_indices]
    X_chi = X_chi[sampled_indices]
    return X_t,X_chi,norm_const


def potential(X_t,M,alpha,T):
    probabilities = np.ones(M)
    for i in range(M):
        if(X_t[i]<T):
            probabilities[i] = np.exp(alpha)
    return probabilities

def initialize(M,N):
    X_t = np.zeros(M)
    X_chi = np.zeros(M)
    norm_consts = np.zeros(N+1)
    return X_t,X_chi,norm_consts

def estimator(M,N,X_chi,norm_consts,params):
    p = np.zeros(N+1)
    for i in range(M):
        p[int(X_chi[i])] = p[int(X_chi[i])] + 1.0
    print p
    p = np.exp(-1.0*params['alpha']*p)
    print p
    normalization = norm_consts.prod()
    print normalization
    p = p * normalization
    return p
