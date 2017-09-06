from __future__ import division

import random
import numpy as np
#import matplotlib.pyplot as plt

def complexSumRules(E, assignDiag=True):
    """Returns complex-valued, off-diagonal position matrix elements x_{nm} that 
    satisfy the diagonal sum rules S_{nn} truncated to N levels. By construction,
    x_{nm} = x_{mn}*.
    
    Parameters
    ----------
    N : int
        -number of states after truncation (including ground state)
        
    INPUT
    E : numpy array object, shape = (NL,)
        -ordered array of normalized energies
        
    RETURNS
    X : numpy complex array, shape = (NL, NL)
        -normalized, complex-valued position matrix
    """
    
    # The number of states
    N = len(E)
    
    # Create complex array to store matrix values in
    # and ensure it's the correct size.
    X = np.zeros((N,N)) + 0j
    assert len(X[0,:]) == N, "Sizes do not match!"
    
    # Index for rows
    i = 0
    
    # We loop through every independent element of the matrix (the upper
    # diagonal elements) and choose a random complex number that satisifies the 
    # constraints.
    while i < N-1:
        # Index for columns
        j = i + 1
        if i is 0:
            # initial cap of entire sum
            Cap = 1
        else:
            # subtract off previously assigned values to get new cap
            Cap = 1 - (E[0:i] - E[i]).dot(abs(X[i,0:i])**2)
            
        #======================================================================
        # I don't know why I put this here.  I think it's incorrect
        '''
        X[i,j]  = np.random.uniform(-1,1) * np.sqrt(Cap) * \
                                    np.exp(1j * np.random.uniform(0,2*np.pi))
        # We want the matrix to be hermitian                        
        X[j,i] = X[i,j].conj()
        '''
        #======================================================================
        
        while j < N:
            if j is N-1:
                # Ensure equality is held
                X[i,j] = np.sqrt(Cap / (E[j] - E[i])) * \
                                    np.exp(1j * np.random.uniform(0,2*np.pi))
                X[j,i] = X[i,j].conj()
            else:
                X[i,j] = np.random.uniform(-1,1) * \
                                    np.sqrt(Cap / (E[j] - E[i])) * \
                                    np.exp(1j * np.random.uniform(0,2*np.pi))
                X[j,i] = X[i,j].conj()
            Cap -= (E[j] - E[i]) * abs(X[i,j])**2
            j += 1
        i += 1
       
    # Assert that sum rules are satisfied 
    # Note: last row will not satisfy sum rules
    k = 0
    while k < len(E) - 1:
        assert np.allclose((E - E[k]).dot(abs(X[k,:])**2), 1), "Diagonal sum rules not satisfied"
        k += 1
        
    if assignDiag is True:
        # If the flag is true, then assign the diagonal elements using the Sigma_{0n}
        # sum rules:
        i = 1
        while i < N-1:    
            X[i,i] = -(np.delete(E - E[i], [0,i]).dot(np.delete(X[i,:]*X[:,0], [0,i]))\
                    + np.delete(E - E[0], [0,i]).dot(np.delete(X[i,:]*X[:,0], [0,i])))\
                    / (E[i] - E[0]) / X[i,0]
            i += 1
        
        # Assert that sum rules are satisfied
        j = 1
        while j < len(E) - 1:
            Sigma = (E - E[j]).dot(X[j,:]*X[:,0]) + (E - E[0]).dot(X[j,:]*X[:,0])
            assert np.allclose(Sigma, 0), "Off-diagonal sum rules not satisfied."
            j += 1

    return X

def realSumRules(E, assignDiag=True):
    """Returns complex-valued, off-diagonal position matrix elements x_{nm} that 
    satisfy the diagonal sum rules S_{nn} truncated to N levels. By construction,
    x_{nm} = x_{mn}*.
    
    Parameters
    ----------
    N : int
        -number of states after truncation (including ground state)
        
    INPUT
    E : numpy array object, shape = (NL,)
        -ordered array of normalized energies
        
    RETURNS
    X : numpy complex array, shape = (NL, NL)
        -normalized, complex-valued position matrix
    """
    
    # The number of states
    N = len(E)
    
    # Create complex array to store matrix values in
    # and ensure it's the correct size.
    X = np.zeros((N,N)) + 0j
    assert len(X[0,:]) == N, "Sizes do not match!"
    
    # Index for rows
    i = 0
    
    # We loop through every independent element of the matrix (the upper
    # diagonal elements) and choose a random complex number that satisifies the 
    # constraints.
    while i < N-1:
        # Index for columns
        j = i + 1
        if i is 0:
            # initial cap of entire sum
            Cap = 1
        else:
            # subtract off previously assigned values to get new cap
            Cap = 1 - (E[0:i] - E[i]).dot(abs(X[i,0:i])**2)
            
        #======================================================================
        # I don't know why I put this here.  I think it's incorrect
        '''
        X[i,j]  = np.random.uniform(-1,1) * np.sqrt(Cap) * \
                                    np.exp(1j * np.random.uniform(0,2*np.pi))
        # We want the matrix to be hermitian                        
        X[j,i] = X[i,j].conj()
        '''
        #======================================================================
        
        while j < N:
            if j is N-1:
                # Ensure equality is held
                X[i,j] = np.sqrt(Cap / (E[j] - E[i])) 
                X[j,i] = X[i,j].conj()
            else:
                X[i,j] = np.random.uniform(-1,1) * \
                                    np.sqrt(Cap / (E[j] - E[i]))
                X[j,i] = X[i,j].conj()
            Cap -= (E[j] - E[i]) * abs(X[i,j])**2
            j += 1
        i += 1
       
    # Assert that sum rules are satisfied 
    # Note: last row will not satisfy sum rules
    k = 0
    while k < len(E) - 1:
        assert np.allclose((E - E[k]).dot(abs(X[k,:])**2), 1), "Diagonal sum rules not satisfied"
        k += 1
        
    if assignDiag is True:
        # If the flag is true, then assign the diagonal elements using the Sigma_{0n}
        # sum rules:
        i = 1
        while i < N-1:    
            X[i,i] = -(np.delete(E - E[i], [0,i]).dot(np.delete(X[i,:]*X[:,0], [0,i]))\
                    + np.delete(E - E[0], [0,i]).dot(np.delete(X[i,:]*X[:,0], [0,i])))\
                    / (E[i] - E[0]) / X[i,0]
            i += 1
        
        # Assert that sum rules are satisfied
        j = 1
        while j < len(E) - 1:
            Sigma = (E - E[j]).dot(X[j,:]*X[:,0]) + (E - E[0]).dot(X[j,:]*X[:,0])
            assert np.allclose(Sigma, 0), "Off-diagonal sum rules not satisfied."
            j += 1

    return X
    
#==============================================================================

def complex_offdiag(E):
    """Returns complex-valued, off-diagonal position matrix elements x_{nm} that 
    satisfy the diagonal sum rules S_{nn} truncated to N levels. By construction,
    x_{nm} = x_{mn}*.
    
    Parameters
    ----------
    N : int
        -number of states after truncation (including ground state)
        
    INPUT
    E : numpy array object, shape = (NL,)
        -ordered array of normalized energies
        
    RETURNS
    X : numpy complex array, shape = (NL, NL)
        -normalized, complex-valued position matrix
    """
    
    # The number of states
    N = len(E)
    
    # Create complex array to store matrix values in
    # and ensure it's the correct size.
    X = np.zeros((N,N)) + 0j
    assert len(X[0,:]) == N, "Sizes do not match!"
    
    # Index for rows
    i = 0
    
    # We loop through every independent element of the matrix (the upper
    # diagonal elements) and choose a random complex number that satisifies the 
    # constraints.
    while i < N-1:
        # Index for columns
        j = i + 1
        if i is 0:
            # initial cap of entire sum
            Cap = 1
        else:
            # subtract off previously assigned values to get new cap
            Cap = 1 - (E[0:i] - E[i]).dot(abs(X[i,0:i])**2)
            
        X[i,j]  = np.random.uniform(-1,1) * np.sqrt(Cap) * \
                                    np.exp(1j * np.random.uniform(0,2*np.pi))
        # We want the matrix to be hermitian                        
        X[j,i] = X[i,j].conj()
        
        while j < N:
            if j is N-1:
                # Ensure equality is held
                X[i,j] = np.sqrt(Cap / (E[j] - E[i])) * \
                                    np.exp(1j * np.random.uniform(0,2*np.pi))
                X[j,i] = X[i,j].conj()
            else:
                X[i,j] = np.random.uniform(-1,1) * \
                                    np.sqrt(Cap / (E[j] - E[i])) * \
                                    np.exp(1j * np.random.uniform(0,2*np.pi))
                X[j,i] = X[i,j].conj()
            Cap -= (E[j] - E[i]) * abs(X[i,j])**2
            j += 1
        i += 1
       
    # Assert that sum rules are satisfied 
    # Note: last row will not satisfy sum rules
    k = 0
    while k < len(E) - 1:
        assert np.allclose((E - E[k]).dot(abs(X[k,:])**2), 1), "Diagonal sum rules not satisfied"
        k += 1
    return X

#==============================================================================

def real_offdiag(E):
    """Returns real-valued, off-diagonal position-matrix elements x_{nm} that 
    satisfy the diagonal sum rules S_{nn} truncated to N levels. By construction,
    x_{nm} = x_{mn}.
    
    Parameters
    ----------
    N : int
        -number of states after truncation (including ground state)
    
    INPUT
    E : numpy array object, shape = (NL,)
        -ordered array of normalized energies
        
    RETURNS
    X : numpy complex array, shape = (NL, NL)
        -normalized, real-valued position matrix
    """
    
    # The number of states
    N = len(E)
    
    # Create complex array to store matrix values in
    # and ensure it's the correct size.
    X = np.zeros((N,N)) + 0j
    assert len(X[0,:]) == N, "Sizes do not match!"
    
    # Index for rows
    i = 0
    
    # We loop through every independent element of the matrix (the upper
    # diagonal elements) and choose a random complex number that satisifies the 
    # constraints.
    while i < N-1:
        # Index for columns
        j = i + 1
        if i is 0:
            Cap = 1
        else:
            Cap = 1 - (E[0:i] - E[i]).dot(abs(X[i,0:i])**2)
        X[i,j]  = np.random.uniform(-1,1) * np.sqrt(Cap) 
        X[j,i] = X[i,j].conj()
        while j < N:
            if j is N-1:
                # Ensure equality is held
                X[i,j] = np.sqrt(Cap / (E[j] - E[i])) 
                X[j,i] = X[i,j].conj()
            else:
                X[i,j] = np.random.uniform(-1,1) * \
                                    np.sqrt(Cap / (E[j] - E[i])) 
                X[j,i] = X[i,j].conj()
            Cap -= (E[j] - E[i]) * abs(X[i,j])**2
            j += 1
        i += 1
       
    # Assert that sum rules are satisfied 
    # Note: last row will NOT satisfy sum rules
       
    k = 0

    while k < len(E) - 1:
        assert np.allclose((E - E[k]).dot(abs(X[k,:])**2), 1), "Sum rules not satisfied for str(k)"
        k += 1
    return X

#==============================================================================
    
def complex_diag(E, X):
    """Returns diagonal position matrix elements x_{qq} that satisfy the 
    off-diagonal sum rules S_{q0}.
    
    Parameters
    ----------
    N : int
        -number of states after truncation (including ground state)
    
    INPUT
    E : numpy array object, shape = (NL,)
        -ordered array of normalized energies
    X : complex numpy array, shape = (NL, NL)
        -complex-valued position matrix
        -all diagonal terms assumed zero
        
    RETURNS
    X : complex numpy array, shape = (NL, NL)
        -normalized, complex-valued position matrix
        -diagonal terms assigned
        
    """
    N = len(E)
    # We pick x_00 to be 1
    X[0,0] = 0.
      
    i = 1
    while i < N-1:    
        X[i,i] = -(np.delete(E - E[i], [0,i]).dot(np.delete(X[i,:]*X[:,0], [0,i]))\
                + np.delete(E - E[0], [0,i]).dot(np.delete(X[i,:]*X[:,0], [0,i])))\
                /(E[i] - E[0])/X[i,0]
        i += 1
                        
    
    # Assert that sum rules are satisfied
    j = 1
    while j < len(E) - 1:
        Sigma = (E - E[j]).dot(X[j,:]*X[:,0]) + (E - E[0]).dot(X[j,:]*X[:,0])
        assert np.allclose(Sigma, 0), "Off-diagonal sum rules not satisfied."
        j += 1
        
    return X

#==============================================================================

def hermitian_offdiag(E):
    """Returns all complex-valued position matrix elements such that 
        1) diagonal and off-diagonal sum rules, S_{nn} and S_{n0} are satisfied 
        2) position matrix is hermitian.
    
    Parameters
    ----------
    N : int
        -number of states after truncation (including ground state)
        
    INPUT
    E : numpy array object, shape = (NL,)
        -ordered array of normalized energies
        
    RETURNS
    X : numpy complex array, shape = (NL, NL)
        -normalized, complex-valued position matrix
        -all elements assigned
    """
    # Number of states
    N = len(E)
    
    # Allocate memory for matrix
    X = np.zeros((N,N)) + 0j
    
    # We first assign the elements x_{n0} = x_{0n}* using S_{00}
    Cap = 1
    n = 1
    
    while n < N:
        if n is N-1:
            # Ensure equality is held
            X[n,0] = np.sqrt(Cap/E[n])\
                    * np.exp(1j * (2*np.pi)*np.random.uniform(0,1))
            X[0,n] = X[n,0].conj()
        else:
            # Assign random complex number with magnitude less than cap
            X[n,0] = np.random.uniform(0,1) * np.sqrt(Cap/E[n])\
                    * np.exp(1j * (2*np.pi)*np.random.uniform(0,1))
            X[0,n] = X[n,0].conj()
        # Increment cap    
        Cap -= E[n]*np.abs(X[n,0])**2
        n += 1
    
    # Assert sum rule S_{00} is satisfied
    assert np.allclose((E*X[0,:]).dot(X[:,0]), 1), "Sum rule S_{00} not satisfied."
    
    X[1,1] = np.random.uniform(-1,1)
    return X
        