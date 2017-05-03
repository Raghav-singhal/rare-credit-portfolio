#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 19:33:26 2017

@author: rsinghal
"""


import numpy as np
from functions import CovMatrix

sigma0 = 0.4
kappa = 3.5
sigmaHat = 0.4
gamma = 0.7
rho_sigma = -0.06

S_0 = 96
r = 0.06
rho = 0.1
B_i = 36
N_sel = 20
dt = 1e-4   # paper has different dt

T1 = 1
T2 = 2
T3 = 3
T4 = 4
T5 = 5

n = 20 # number of selection steps in a year
alpha = 0.1 # experimental parameter

timesteps = int(1./(n*dt)) # T/n*dt where n = 20 per year 


"""
Xn = [sigma(n),S1,...,SN,min(S1,...,SN)]

euler scheme --- Xn+1 = Xn + a(Xn)dt + b(Xn)dW

a(Xn) = [kappa*(sigmaHat - sigma(n)) ,r*S1,..,r*SN]  
b(Xn) =  [gamma*np.sqrt(sigma(n)),simga0*sigma(n)*S1,.....,simga0*sigma(n)*SN]

portfolio = [sigma,S1,...SN]
"""
def a(portfolio):
    sigma,S = portfolio[0],portfolio[1:]
    
    return np.vstack((kappa*(sigmaHat - sigma),r*S))

def b(portfolio):
    sigma,S = portfolio[0],portfolio[1:]
    
    return np.vstack((gamma*np.sqrt(sigma),sigma0*sigma*S))

"""
X = M * number of portfolios * 1
"""
def func(X):
    return [a(x) for x in X],[b(x) for x in X]

def Mutation(Xn,T):
    M = np.shape(Xn)[0] # number of portfolios
    N = int((np.shape(Xn)[1]-1)/2) # number of assets in a portfolio
    
    C = CovMatrix(N,dt)
    
    X = [(x[:N+1,:]) for x in Xn] 
    minX = [(x[N+1:,:]) for x in Xn]
            
    for i in range(T*timesteps):
    
        a1,b1 = func(X)
        
        dW = np.sqrt(dt)*np.random.multivariate_normal(np.zeros(N+1),C,M)
        
        for j in range(M):
            X[j] = X[j]+ a1[j]*dt + b1[j]*dW[j].reshape(N+1,1)
            
    
    Xhat = []
    for j in range(M):
        for i in range(1,N+1):
            if X[j][i,0] < minX[j][i-1,0]:
                minX[j][i-1,0] = X[j][i,0]
        Xhat.append(np.vstack((X[j],minX[j])))
    
    return Xhat,Xn
    
    

    
