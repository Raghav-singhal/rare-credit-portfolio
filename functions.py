#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 15:55:06 2017

@author: rsinghal
"""

import numpy as np
from scipy.stats import norm
from scipy.stats import rv_discrete
import matplotlib.pyplot as plt

sigma0 = 0.5
kappa = 3.5
sigmaH = .4
gamma = .7
rho_sigma = -0.06

S_0 = 90
r = 0.06
rho = 0.1
B_i = 36
N_sel = 20
dt = 1e-3

T1 = 1
T2 = 2
T3 = 3
T4 = 4
T5 = 5

N = 125

alpha = 0.1 # experimental parameter
"""
C = (N+1)*(N+1)

Produces the covariance matrix for the brownian motion time step (dt)

row and col 0 are the correlations for the stochastic volatility process

"""
def CovMatrix(N,dt):
    N = N + 1
    C = np.zeros((N,N))
    
    for i in range(N):
        for j in range(i,N):
            
            if j==i: 
                C[i,j] = dt
            elif i==0 or j==0: 
                C[i,j] = rho_sigma*dt
                C[j,i] = C[i,j]
            else:
                C[i,j] = rho*dt
                C[j,i] = C[i,j]
    print(C)
    return C
    
"""

Generate N independent samples given a DISCRETE empirical DENSITY function

"""
def empirical_sampler(N,dist,x):

    arr = rv_discrete(values = (x,dist)).rvs(size=N)
            
    print("Actual Mean is %s"%np.dot(dist,x))
    print("Empirical Mean is %s"%np.mean(arr))
    
    return arr

x = [0,1,3,4,5,6]
dist = [0.5,0.1,0.1,0.1,0.1,0.1]

a = empirical_sampler(10000,dist,x)
    
    
    
    
        
