# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 17:01:19 2018

@author: Ethan
"""

import numpy as np
import scipy as sp
import scipy.linalg
import heapq as hq

def sum_rules(E, x):
    """Return sum rule matrix, given eigenenergies and transition moments.
    It's assumed that the dipole moments have been assigned.
    
    Input
        E : np.array
            eigenenergies
        x : np.array
            transition moments
            
    Output
        SR : np.array
            sum rule matrix: i.e. SR[p,q] corresponds to the (p,q) sum rule.
    """
    
    # The hamiltonian is diagonal
    H = np.diag(E)
    
    SR = x.dot(x.dot(H)) - 2*x.dot(H.dot(x)) + H.dot(x.dot(x))
    
    return SR


def get_S(N):
    """Returns a random, anti-symmetric array such that each row sums to 1.
    
    Input 
        N : int
            desired number of states
        
    Output
        S : array
            sum rule matrix
    """

    T = np.random.random((N, N))  # get random array
    T = sp.linalg.triu(T, k=1)    # Make elements on diagonal and below zero
    r = T.sum(axis=-1)[:-1]       # Make a N-1 x 1 array where each element
                                  # is the sum of a row in T excluding the
                                  # last row, whcih cannot have a positive sum

    
    M = np.diag(r) - T[:-1, :-1]
    # Keeping M upper triangular is faster.
    d = sp.linalg.solve_triangular(M, trans=True, b=np.ones(N-1
    ), overwrite_b=True, check_finite=False)

    T[:-1,:] *= d[:,None]
    A = T - T.T
    
    return A


def get_Enm_Xnm(S):
    """Returns  transition moments.
    
    Input
        S : np.array
            sum rule matrix
        Enorm : np.array
            normalized eigenenergies.
            
    Output
        Xnm : np.array
            array of transition moments
    """
    N = len(S[0])
    E = np.random.random((N))       # pick randome energies
    EnOrder = hq.nsmallest(N,E)     # order the energies
    Enorm = (EnOrder - EnOrder[0])/(EnOrder[1] - EnOrder[0])
                                    # normalize energies
                                    
    Enm = np.outer(Enorm,np.ones(N)) - np.outer(np.ones(N),Enorm)
                                    # Calculate energy matrix E_nm = E_n - E_m
                                    
    Enm = Enm + np.diag(np.ones(N))
                                    # Place ones on the diagonal to avoid
                                    # division by zero
        
    RandomSign = np.sign(np.random.random((N,N))-.5)
                                    # Make matrix of random +1s and -1s
    RandomSign = ( sp.linalg.triu(RandomSign, k=0)
    + np.transpose(sp.linalg.triu(RandomSign, k=1)) )                        
                                    # Enforce symmetry
    
    x =  RandomSign*np.sqrt(abs(S/Enm))
                                    # Determine trasition moment matrix
    
    H = np.diag(Enorm)              # Hamltonian matrix is diagonal
    
    xnn = np.dot(2.*np.dot(x,H)-np.dot(H,x),x[:,0])[1:] / x[1:,0] / Enorm[1:]
                                    # Use sum rules to get dipole moments
                                    
    x = x - np.diag(np.append(0, xnn))
                                    # Add dipole moments to diagonal part of x 
                                    # By convention we let x[0,0] = 0.
    
    return Enorm, x

def sigma(SR, N):
    """Returns standard deviation of the sum rules. The standard deviation
    is computed as
        sigma = 2 Norm(SR_tilde) / N / (N+1)
    where Norm() is the Frobenius norm and SR_tilde is the upper-triangular
    part of the sum rule matrix.
    
    Input
        SR : np.array
            sum rule matrix
        N : np.int
            size of the square submatrix of SR, SR_tilde, to use in computing 
            sigma.
        
    Output
        sigma : np.float
            standard deviation of sum rule matrix.
    """
    
    # Truncate SR matrix to NxN, upper-triangular submatrix
    SR_tilde = sp.linalg.triu(SR[:N,:N], k=1)
    
    # Standard deviation as defined in lytel17.01
    sigma = 2 * sp.linalg.norm(SR_tilde) / N / (N + 1)
    
    return sigma

