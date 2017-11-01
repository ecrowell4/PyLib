# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 19:13:55 2017

@author: Ethan Crowell

This file contains the SOS expressions for the magnetic polarizabilities and
hyperpolarizabilities.  Specifically, it contains the coefficients describing
the magnetic dipole moment induced by a monochromatic magnetic field:
    m = m0 + alpha B + beta B^2 + gamma B^3 + ...
"""

import numpy as np
from numba import jit

def alpha_mm(E, L, I, omega=0, damp=0, canonical=False, N=None):
    """Returns the canonical, linear, magnetic polarizability alpha at input frequency
    omega:
        m = alpha * B(oemga)
    
    INPUT
    -----
    E : np.array(numstates)
        eigenenergies of unperturbed system
    L : np.array((numstates, numstates))
        angular momentum matrix of unperturbed system
    I : complex
        unperturbed, ground state expectation value of moment of inertia operator: <0|I|0>
    omega : float
        frequency of incident magnetic perturbation
    damp: float
        damping constant for system, assumed to be zero unless otherwise specified
        
    OUTPUT
    ------
    alpha : complex
        magnetic polarizability
    """
    
    # Fundamental constants (Hartree units)
    hbar = 1
    e = 1
    m = 1
    c = 137.06
    
    # Gyromagnetic ratio for electron
    g = e / 2 / m / c
    
    # If desired number of states to include in SOS isn't specified, use all
    # of the states given
    if N is None:
         N = len(E)
        
    # Evaluate the SOS term in equation for alpha^{mm}
    alpha = g**2 * (L[0,1:N].dot((L[1:N,0] / (E[1:N]- E[0] - hbar*omega - 1j*hbar*damp)))
                       + L[0,1:N].dot((L[1:N,0] / (E[1:N]-E[0] + hbar*omega + 1j*hbar*damp))))
    
    # Include the diamagnetic term (i.e. Faraday term) if the user doesn't
    # specify not to
    if canonical is True:
        return alpha
    else:
        return alpha - g**2 * I
    
def beta_mmm(E, L, I, omega1=0, omega2=0, damp=0, canonical=False):
    """Returns the canonical, diagonal tensor component of the quadratic mangetic polarizability 
    beta at imput frequencies omega1 and omega2:
        m = beta B(omega1) B(omega2)
        
    INPUT
    -----
    E : np.array(numstates)
        eigenenergies of unperturbed system
    L : np.array((numstates,numstates))
        angular momentum matrix of unperturbed system
    I : np.array((numstates, numstates))
        moment of inertia matrix of unperturbed system
    omega1, omega2 : floats
        frequencies of incident magnetic perturbations
    damp : float
        damping constant for system; assumed to be zero unless otherwise specified
    
    OUTPUT
    ------
    beta : comples
        first magnetic hyperpolarizability
    """
    
    # Constants
    hbar = 1
    e = 1
    m = 1
    c = 137.036
    gamma = e / 2 / m / c
    
    beta =  0.5 * gamma**3 * (
    (L[0,1:]/(E[1:] - E[0] - hbar*omega1 - hbar*omega2 - 1j*hbar*damp)).dot(L[1:,1:].dot(L[1:,0] / (E[1:] - E[0] - hbar*omega2 - 1j*hbar*damp))) 
    + (L[0,1:]/(E[1:] - E[0] - hbar*omega1 - hbar*omega2 - 1j*hbar*damp)).dot(L[1:,1:].dot(L[1:,0] / (E[1:] - E[0] - hbar*omega1 - 1j*hbar*damp)))
    + (L[0,1:]/(E[1:] - E[0] + hbar*omega1 + 1j*hbar*damp)).dot(L[1:,1:].dot(L[1:,0] / (E[1:] - E[0] - hbar*omega2 - 1j*hbar*damp)))
    + (L[0,1:]/(E[1:] - E[0] + hbar*omega2 + 1j*hbar*damp)).dot(L[1:,1:].dot(L[1:,0] / (E[1:] - E[0] - hbar*omega1 - 1j*hbar*damp)))
    + (L[0,1:]/(E[1:] - E[0] + hbar*omega1 + hbar*omega2 + 1j*hbar*damp)).dot(L[1:,1:].dot(L[1:,0] / (E[1:] - E[0] + hbar*omega2 + 1j*hbar*damp)))
    + (L[0,1:]/(E[1:] - E[0] + hbar*omega1 + hbar*omega2 + 1j*hbar*damp)).dot(L[1:,1:].dot(L[1:,0] / (E[1:] - E[0] + hbar*omega1 + 1j*hbar*damp)))
    ) 
    
    if canonical is True:
        return beta
    else:
        beta
    return beta + 0.5 * gamma**3 * (
            I[0,1:].dot(L[1:,0] / (E[1:] - E[0] - hbar*omega1 - 1j*hbar*damp)) 
            + I[0,1:].dot(L[1:,0] / (E[1:] - E[0] - hbar*omega2 - 1j*hbar*damp))
            + L[0,1:].dot(I[1:,0] / (E[1:] - E[0] + hbar*omega1 + 1j*hbar*damp))
            + L[0,1:].dot(I[1:,0] / (E[1:] - E[0] + hbar*omega2 + 1j*hbar*damp)))
    

@jit    
def gamma_mmmm(E, L, I, omega1=0, omega2=0, omega3=0, damp=0, canonical=False, N=None):
    """Returns the canonical, diagonal tensor component of the quadratic mangetic polarizability 
    beta at imput frequencies omega1 and omega2:
        m = beta B(omega1) B(omega2)
        
    INPUT
    -----
    E : np.array(numstates)
        eigenenergies of unperturbed system
    L : np.array((numstates,numstates))
        angular momentum matrix of unperturbed system
    I : np.array((numstates, numstates))
        moment of inertia matrix of unperturbed system
    omega1, omega2 : floats
        frequencies of incident magnetic perturbations
    damp : float
        damping constant for system; assumed to be zero unless otherwise specified
    
    OUTPUT
    ------
    Gamma : comples
        first magnetic hyperpolarizability
    """
    
    # Constants
    hbar = 1
    e = 1
    m = 1
    c = 137.036
    gamma = e / 2 / m / c
    if N is None:
        N=len(E)
    Gamma =  0
    i = 1
    j = 1
    l = 1
    Lbar = L - L[0,0]
    while i < N:
        while j < N:
            while l < N:
                Gamma += L[0,i]*Lbar[i,j]*Lbar[j,l]*L[l,0] / ((E[i]-E[0])*(E[j]-E[0])*(E[l]-E[0]))
                l += 1   
            j += 1
        i += 1
    
    while i < N:
        while j < N:
            Gamma -= L[0,i]*L[i,0]*L[0,j]*L[j,0] / ((E[j]-E[0])**2 * (E[i]-E[0]))
    
    Gamma *= 4*gamma**4 / hbar**3
    
    if canonical is True:
        return Gamma
    else:
        i = 1
        j = 1
        while i < N:
            while j < N:
                if i != j:
                    Gamma += (2*I*gamma**4 / hbar**2) * L[0,i]*L[j,0] / ((E[i]-E[0])*(E[j]-E[0]))
                j += 1
            i += 1
        return Gamma
    
def Rotation(E, xx, L, omega):
    """Computes Rosenfelds SOS expression for the optical rotation."""
    
    e = 1
    m = 1
    c = 137.036
    hbar = 1

    return (e**2 * hbar / 12 / m / c) * omega**2 * xx[0,1:].dot(L[1:,0] / 
             ((E[1:] - E[0])**2 - omega**2))