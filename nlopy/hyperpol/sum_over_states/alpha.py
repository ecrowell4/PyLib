# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 12:38:46 2017

@author: Ethan Crowell
"""

import numpy as np
from nlopy.hyperpol.sum_over_states import sos_utils as sos

    
def alpha_ee(E, XX, units, omega=0):
    """Calculates the polarizability alpha as a function of the energy spectrum
    and transition matrix. The SOS expression derived from time-dependent
    perturbation theory is evaluated using the first num_states states.
    
    Input
        E : np.array(N)
            energy spectrum
        XX : list [xx_exp, xx_pert]
            List whose elements are the transition matrices that enter into SOS
            due to expection value and perturbation, respectively. These are 
            typically the same except for when quasi degenerate methods are
            implemented.
        units : class
            class whose attributes are the fundamental constants hbar, e, m, c, etc.
        omega : float, default value of zero
            frequency of incident electric field
        
    Output
        alpha : complex
            polarizability describing the linear response of the polarization
            to an electric field.
    """
    
    # Determine number of states in input:
    num_states = len(E)
    
    # We assume that transition moments that are np.allclose to zero actually vanish,
    #   and are only nonzero due to numerical errors:
    if np.allclose(XX[1][1,0], 0):
        start = 2
    else:
        start = 1
            
    return units.e**2 * (XX[0][0,start:].dot(XX[1][start:,0] * sos.D1(E, omega, units, start)) 
        + XX[1][0,start:].dot(XX[0][start:,0] * sos.D1(E.conjugate(), -omega, units, start)))
    
def alpha_quasi_degen(E_prime, xx_prime, E10, coeff_diff):
    E_denom = 2*(E_prime[2:] - E_prime[0]) - E10*coeff_diff
    
    return 4 * xx_prime[0,2:].dot(xx_prime[2:,0] / E_denom)
