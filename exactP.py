#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 19:42:39 2017

@author: rsinghal
"""

import numpy as np
from scipy.stats import norm


def P_exact(ratio,T,r,sigma):
    # ratio = S_0/B
    p = 1 - 2*r/sigma**2
    
    dp = (np.log(ratio) + (r - 0.5*sigma**2 )*T)/(sigma*np.sqrt(T))
    dn = (-np.log(ratio) + (r - 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    
    return 1 - (norm(0,1).cdf(dp) - (ratio**p)*norm(0,1).cdf(dn))
    
    