#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 19:33:26 2017

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
    return C

dt = 1e-4   # paper has different dt

n = 20 # number of selection steps in a year

timesteps = int(1./(n*dt)) # number of timesteps in a year

"""
Xn = [sigma(n),S1,...,SN,min(S1,...,SN)]

euler scheme --- Xn+1 = Xn + a(Xn)dt + b(Xn)dW

a(Xn) = [kappa*(sigmaHat - sigma(n)) ,r*S1,..,r*SN]  
b(Xn) =  [gamma*np.sqrt(sigma(n)),simga0*sigma(n)*S1,.....,simga0*sigma(n)*SN]

portfolio = [sigma,S1,...SN]
"""
def a(portfolio):
    sigma,S = portfolio[0],portfolio[1:]
    
    return np.hstack((kappa*(sigmaHat - sigma),r*S))

def b(portfolio):
    sigma,S = portfolio[0],portfolio[1:]
    
    return np.hstack((gamma*np.sqrt(sigma),sigma0*sigma*S))

"""
input Xn =  M copies of [sigma(n),S1,...,SN,min(S1,...,SN)]
output Xn+1 , Wn+1 

"""
def func(X):
    return np.asarray([a(x) for x in X]),np.asarray([b(x) for x in X])

def Mutation(Xn,T,dt,n):
    M = np.shape(Xn)[0] # number of portfolios
    N = int((np.shape(Xn)[1]-1)/2) # number of assets in a portfolio
    
    C = CovMatrix(N,dt)
    
    timesteps = int(1/(n*dt)) #number of timesteps
    X = Xn[:,:N+1] 
    minX = Xn[:,N+1:]
    
    for i in range(timesteps):
        a1,b1 = func(X)
        dW = np.sqrt(dt)*np.random.multivariate_normal(np.zeros(N+1),C,M)
        
        for j in range(M):
            X[j] = X[j] + a1[j]*dt + b1[j]*dW[j]
            
    Xhat = []
    for i in range(M):
        for j in range(1,N+1):
            if X[i,j] < minX[i,j-1]:
                minX[i,j-1] = X[i,j]
                
        Xhat.append(np.hstack((X[i,:],minX[i,:])))

    return np.asarray(Xhat)

