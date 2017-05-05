#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 15:55:10 2017

@author: rsinghal
"""
import numpy as np
sigma0 = 0.4
kappa = 3.5
sigmaHat = 0.4
gamma = 0.7
rho_sigma = -0.06

S_0 = 90
r = 0.06
rho = 0.1


def lowerChol(N):
    N += 1
    C = np.zeros((N,N))
    
    for i in range(N):
        for j in range(i,N):    
            if j==i: 
                C[i,j] = 1
            elif i==0 or j==0: 
                C[i,j] = rho_sigma*1
                C[j,i] = C[i,j]
            else:
                C[i,j] = rho*1
                C[j,i] = C[i,j]
    
    lower=np.linalg.cholesky(C)
    return C,lower


"""
return correlated brownian motion at time T, and the covariance
"""
def br(N,T, samples=10000, dt=0.1):
    
    C,lower = lowerChol(N)
    
    dW=0
    for i in range(int(T/dt)):
        Z = np.sqrt(dt)*np.random.multivariate_normal(np.zeros(N+1),np.eye(N+1),samples)
        
        Z = np.dot(Z,lower)
    
        dW += Z
    
    dW0 = dW[:,0]
    dW1 = dW[:,1]
    dW2 = dW[:,2]
    
    print(np.cov(dW1,dW2),"\n")
    print (np.cov(dW0,dW1))
    
    return dW

br(2,1)
