# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 17:01:19 2018

@author: Ethan
"""

import numpy as np
import scipy as sp
import scipy.linalg
import heapq as hq

def sum_rules(E, x, normalize=False):
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
        normalize : bool
            if True, then use scaled quantities xi_nm = x_nm / x_max 
            and e_nm = E_nm / E_10
        fix_E10: bool
            if True, then E10 is fixed
        fix
        
    Output
        S : array
            sum rule matrix
        normalzie : bool
            same as input normalize. Will be used in later functions,
            ensuring consistency regardless of users stupidity.
    """

    T = np.random.random((N, N))  # get random array
    T = sp.linalg.triu(T, k=1)    # Make elements on diagonal and below zero
    r = T.sum(axis=-1)[:-1]       # Make a N-1 x 1 array where each element
                                  # is the sum of a row in T excluding the
                                  # last row, whcih cannot have a positive sum

    
    M = np.diag(r) - T[:-1, :-1]
    # Keeping M upper triangular is faster.

    if normalize is True:
        b_vec = np.ones(N-1)
    else:
        b_vec = 0.5 * np.ones(N-1)

    # find vector d such that (T*d) - (T*d).transpose() obeys sum rules
    d = sp.linalg.solve_triangular(M, trans=True, b=b_vec, overwrite_b=True,
        check_finite=False)

    T[:-1,:] *= d[:,None]
    A = T - T.T
    
    return A


def get_Enm_Xnm(S, spectrum=None):
    """Returns  transition moments.
    
    Input
        S : np.array
            sum rule matrix
        spectrum : np.array
            Normalized spectrum of energies:
            E[n] = f(n) where 
            If None, then a random spectrum is chosen
            Example : quadratic spectrum f(n) = n**2: [0,1,4,9,...]
                      HO spectrum f(n) = n: [0,1,2,3,...]
                      sub linear f(n) = sqrt{n}: [0,1,sqrt{2},sqrt{3},...] 
            
    Output
        E_ordered : np.array
            ordered array of eigenenergies normalized
            by E10
        Xnm : np.array
            array of transition moments normalized by x_max
    """
    normalize = S[0]
    N = len(S[1][0])
    if spectrum is None:
        E = np.random.random((N))       # pick random energies
        E_ordered = hq.nsmallest(N,E)     # order the energies
        E_ordered = E_ordered - E_ordered[0] # grnd state is zero energy
    else:
    	E = spectrum    # Assume input spectrum is properly formatted

    # Normalized spectrum by E10 by E10
    E_ordered = (E_ordered - E_ordered[0]) / (E_ordered[1] - E_ordered[0])
                                    # normalize energies
                                    
    Enm = np.outer(E_ordered,np.ones(N)) - np.outer(np.ones(N),E_ordered)
                                    # Calculate energy matrix E_nm = E_n - E_m
                                    
    Enm = Enm + np.diag(np.ones(N))
                                    # Place ones on the diagonal to avoid
                                    # division by zero
        
    RandomSign = np.sign(np.random.random((N,N))-.5)
                                    # Make matrix of random +1s and -1s
    RandomSign = ( sp.linalg.triu(RandomSign, k=0)
    + np.transpose(sp.linalg.triu(RandomSign, k=1)) )                        
                                    # Enforce symmetry
    
    x =  RandomSign*np.sqrt(abs(S[1] / Enm))
                                    # Determine trasition moment matrix
    
    H = np.diag(E_ordered)              # Hamltonian matrix is diagonal
    
    xnn = np.dot(2.*np.dot(x,H)-np.dot(H,x),x[:,0])[1:] / x[1:,0] / E_ordered[1:]
                                    # Use sum rules to get dipole moments
                                    
    x = x - np.diag(np.append(0, xnn))
                                    # Add dipole moments to diagonal part of x 
                                    # By convention we let x[0,0] = 0.
    
    return E_ordered, x

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

def get_rand_unitary(N):
    """Returns a random unitary matrix of the desired size.
    The matrix is of the form exp(iA), where A is a hermitian
    matrix.

    Input
        N : int
            dimension of matrix

    Output
        U : np.array
            unitary matrix
    """

    # Generate random matrix and make it hermitian
    A = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    A = -.5 * (A + A.transpose().conjugate())

    # Create unitary matrix from A
    U = sp.linalg.expm(1j * A)

    return U

