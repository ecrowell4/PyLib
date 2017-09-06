# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 12:38:46 2017

@author: Ethan Crowell
"""

import numpy as np

def D1(E, omega):
    """Returns the propogator for a first order process.
    Args
    ----
    E : np.array
        array of eigenergies.
        Note: if omega is take as negative, then user should pass E.conjugate() instead
        of E.
    omega : float
        input frequency"""
    
    # Take E -> E - E0
    E = E - E[0]
    
    # Compute and return the propogator
    return 1 / (E[1:] - omega)

def D2(E, omega1, omega2):
    """Returns propogator for a second order process."""
    E = E - E[0]
    return 1 / (E[1:] - omega1 - omega2)


    
def alpha_tst(E, xx, omega=0):
    return xx[0,1:].dot(xx[1:,0] * D1(E, omega)) + xx[0,1:].dot(xx[1:,0] * D1(E.conjugate(), -omega))

def alpha_ee(E, xx, omega=0, damping=False, num_states=None):
    """Calculates the polarizability alpha as a function of the energy spectrum
    and transition matrix. The SOS expression derived from time-dependent
    perturbation theory is evaluated using the first num_states states.
    Input
    -----
    E : np.array(N)
        energy spectrum
    xx : np.array((N,N))
        transition matrix
    Optional
    --------
    omega : float
        frequency of incident electric field
    damping: boolean
        if true, then damping factor is defined using Fermi's golden rule, otherwise
        it's zero.
    num_states : integer
        number of states included in the SOS. By default, the script uses the entire
        spectrum given it.
    
    Output
    ------
    alpha : complex
        polarizability describing the linear response of the polarization
        to an electric field.
    
    References
    ----------
    [1] : Mark Kuzyk, Fundamental Limits of all nonlinear-optical phenomena that
    that are representable by a second-order nonlinear susceptibility, J Chem Phys,
    125, 145108 (2006).
    """
    
    # Fundamental constants
    e = 1
    hbar = 1
    c = 137.036
    
    # If the desired number of states isn't specified, then use all states given
    if num_states is None:
        num_states = len(E)

    if damping == True:
        # Use natural linewidth: see reference [1].
        Gamma = (2 / 3) * ((E[0] - E[1:num_states+1]) / hbar / c)**3 * abs(xx[0,1:num_states])**2
    else:
        Gamma = 0
        
    # Evaluate SOS expression for alpha
    alpha = e**2 * (xx[0,1:num_states+1].dot(xx[1:num_states+1, 0] / (E[1:num_states+1]-E[0] - 1j*Gamma - hbar*omega))
    + xx[0,1:num_states+1].dot(xx[1:num_states+1, 0] / (E[1:num_states+1]-E[0] + 1j*Gamma + hbar*omega)))
    
    return alpha

#==============================================================================
# This function was defined to test that dot products in previous function were
# one properly.
def alpha_sum(E, xx, num_states=None):
    e = 1
    hbar = 1
    
    if num_states is None:
        num_states = len(E)
        
    alpha=0
    i = 1
    while i < num_states:
        alpha += xx[0,i]*xx[i,0] / (E[i] - E[0])
        i += 1
    
    return alpha
    