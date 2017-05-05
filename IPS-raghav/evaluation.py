#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 22:18:50 2017

@author: rsinghal
"""
import numpy as np

def potential(Xp, Wp, alpha):
    numPortfolios = Xp.shape[0]
    n = int((Xp.shape[1] - 1) / 2)
    min_Xp = Xp[:, -n:]
    min_Wp = Wp[:, -n:]
    VXp = np.sum(np.log(min_Xp), axis=1)
    VWp = np.sum(np.log(min_Wp), axis=1)
    G = np.exp(-alpha * (VXp - VWp))
    
    return G

            
"""
input = M portfolios , Barrier price
output = number of assets below Barrier price
"""
    
def f(Xn,B):
    N = int((np.shape(Xn)[0]-1)/2)
    minX = Xn[N+1:]
    k = 0
    
    for x in minX:
        if x < B: k+=1

    return k
    
"""

"""

def p_k(X0,Wn,B,alpha,norm_const):
    N = int((np.shape(X0)[1]-1)/2.)
    M = X0.shape[0]

    p = np.zeros(N+1)
    
    particle_defaults = [f(y,B) for y in X0]
    G = potential(X0,Wn,alpha)
    
    for k in range(N+1):
        for j in range(M):
            if particle_defaults[j] == k:
                p[k] +=  G[j]/M
    p = p*norm_const
                
    return p

M = 1
N = 3

X = np.ones((M,2*N+1))

for i in range(M):
    for j in range(2*N+1):
        if j==N+0:
            X[i,j] = 1.04
        elif j==N:
            X[i,j] = 40.0
        elif j==N+1:
            X[i,j] = 99.0
            
print(p_k(X,X,50,0,1))
          
