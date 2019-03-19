# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 12:38:46 2017

@author: Ethan Crowell
"""

import numpy as np
from nlopy.hyperpol.sum_over_states import sos_utils

def alpha_ee(E, xx, units, omega=0, intrinsic=False, n=0, damping=False):
    """Calculates the polarizability alpha for a system in the ground
    
    Input
        E : np.array(N)
            energy spectrum
        xx : np.array((N,N))
            transition matrix
        units : class
            class object containing fundamental constants
        omega : float
            frequency of incident electric field
        intrinsic : bool
            if True, then intrinsic alpha is returned. In this case
            it is assumed the scaled xx and E are given as input
        n : int
            state system is assumes to be in (i.e. n=0 -> ground state)
        damping : bool
            If true, then include natural linewidth as damping term
            
        
    Output
        alpha : complex
            electric polarizability
    """

    # assert consistent dimensions
    assert len(E)==len(xx[0]), "dimensions of E and xx do not match."
    
    # Take all Em -> Emn
    E = E - E[n]

    if damping == True:
        E = E - 1j * (2 / 3 / units.hbar) * (E / units.hbar / units.c)**3 * units.e**2 * abs(xx[:,0])**2

    
    alpha =  units.e**2 * (np.delete(xx[n,:], n).dot(np.delete(xx[:,n], n) * sos_utils.D1(np.delete(E, n), omega, units))
                           + np.delete(xx[n,:], n).dot(np.delete(xx[:,n], n) * sos_utils.D1(np.delete(E.conjugate(), n), -omega, units))
                          )

    if intrinsic is True:
        return alpha / 2
    else:
        return alpha


def alpha_mm(E, L, I, units, omega=0, canonical=False):
    """Returns the canonical, linear, magnetic polarizability alpha at input frequency
    omega:
        m = alpha * B(oemga)
    
    INPUT
        E : np.array
            eigenenergies
        L : np.array
            angular momentum matrix
        I : np.complex
            ground state expectation value of moment of inertia operator: <0|I|0>
        units : Class
            class object containing fundamental constants
        omega : float
            frequency of incident magnetic perturbation
        canonical : bool
            if True, return alpha using canonical angular momentum, not mechanical
        
    OUTPUT
        alpha : complex
            magnetic polarizability
    """
    
    g = units.e / 2 / units.m / units.c
        
    # Evaluate the SOS term in equation for alpha^{mm}
    alpha_mag = units.g**2 * (L[0,1:].dot((L[1:,0] * sos_utils.D1(np.delete(E, 1), omega, units)))
                       + L[0,1:].dot((L[1:,0] * sos_utils.D1(np.delete(E.conjugate(), 1), omega, units))))
    
    # Include the diamagnetic term (i.e. Faraday term) if the user doesn't
    # specify not to
    if canonical is True:
        return alpha_mag
    else:
        return alpha_mag - g**2 * I



