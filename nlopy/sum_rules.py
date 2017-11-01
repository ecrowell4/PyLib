# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 09:53:40 2017

@author: Owner
"""

import numpy as np
# Import my personal library.
import sys
sys.path.append(r'C:\Users\Owner\Dropbox\PyLib')    # path for my machine
sys.path.append('..')   # path for Kamiak
import PyLib as pl

def sum_rules(E, xx, N=None, p=None, q = None):
    """Returns the subset of sum rules Snm for which n < p and m < q.
    Paramters
    ---------
    E : np.array(N)
        eigenenergies
    xx : np.array(N,N)
        transition matrix
    N : int, optional
        number of state to include in each sum. If None, then use all states
        passed.
    p : int, optional
        largest row of sum rule matrix to compute. Must be less than N. Default
        is N
    q : int, optional
        largest column of sum rule matrix to compute. Must be less than N. Default
        is N.
        
    Returns
    -------
    sum_rules : np.array(p, q)
        sum rule matrix
    """
    
    # Set default values of unspecified optional variables:
    if N is None:
        N = len(E)
    if p is None:
        p = N
    if q is None:
        q = N
        
    # We first assert that we have all the states to evaluate sum rules
    assert p <= N, 'Do not have enough states to evaulate '+str(p)+'th row of sum rules.'
    assert q <= N, 'Do not have enough states to evaulate '+str(q)+'th row of sum rules.'
    
    sum_rules = -0.5 * (xx.dot(xx.dot(np.diag(E))) - 2*xx.dot(np.diag(E).dot(xx))
    + np.diag(E).dot(xx.dot(xx)))

    return sum_rules
