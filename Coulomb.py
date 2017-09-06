"""
Created on Wed Mar 29 10:58:54 2017

@author: Ethan Crowell

This script contains my first attempt at solving the SE for two interacting 
electrons.  I assume the electrons are confined to a box with infinite walls.  
I will allow from some external potential V(x,y) to be inside the box.
"""

import numpy as np
def Coulomb(x, y, q1, q2):
    """Returns an array W that represents the Coulomb interaction energy between
    two charges:
        Wij = q1 * q2 / |r1 - r2|.
    To avoid divergences, we set the diagonal elements to some finite but larege
    value.
    
    INPUT
    -----
    x, y : np.array
        two arrays representing the position space of the problem
    q1, q2 : floats
        the charges of the two particles
    
    OUTPUT
    ------
    W : np.array
        interaction energy
    """
    
    N = len(x)
    dx = x[1]-x[0]
    
    W = np.zeros((N, N))
    
    r = np.sqrt(x**2 + y**2)

    for i in range(N):
        for j in range(N):
            if i == j:
                if i == N-1:
                    W[i,i] = q1 * q2 / abs(r[i] - r[i-1]) + dx * q1 * q2 / abs(r[i] - r[i-1])**2
                else:
                    W[i,i] = q1 * q2 / abs(r[i] - r[i+1]) + dx * q1 * q2 / abs(r[i] - r[i+1])**2
            else:
                W[i,j] = q1 * q2 / abs(r[i] - r[j])
    return W

from numba import jit

@jit
def jCoulomb(x, y, q1, q2):
    """Returns an array W that represents the Coulomb interaction energy between
    two charges:
        Wij = q1 * q2 / |r1 - r2|.
    To avoid divergences, we set the diagonal elements to some finite but larege
    value.
    
    INPUT
    -----
    x, y : np.array
        two arrays representing the position space of the problem
    q1, q2 : floats
        the charges of the two particles
    
    OUTPUT
    ------
    W : np.array
        interaction energy
    """
    
    N = len(x)
    dx = x[1]-x[0]
    
    W = np.zeros((N, N))
    
    r = np.sqrt(x**2 + y**2)

    for i in range(N):
        for j in range(N):
            if i == j:
                if i == N-1:
                    W[i,i] = q1 * q2 / abs(r[i] - r[i-1]) + dx * q1 * q2 / abs(r[i] - r[i-1])**2
                else:
                    W[i,i] = q1 * q2 / abs(r[i] - r[i+1]) + dx * q1 * q2 / abs(r[i] - r[i+1])**2
            else:
                W[i,j] = q1 * q2 / abs(r[i] - r[j])
    return W