# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 12:38:46 2017

@author: Ethan Crowell
"""

import numpy as np
from nlopy.hyperpol.sum_over_states import sos_utils as sos

    
def alpha_ee(E, xx, units, omega=0):
    """Calculates the polarizability alpha as a function of the energy spectrum
    and transition matrix. The SOS expression derived from time-dependent
    perturbation theory is evaluated using the first num_states states.
    
    Input
        E : np.array(N)
            energy spectrum
        xx : np.array((N,N))
            transition matrix
        units : class
            class whose attributes are the fundamental constants hbar, e, m, c, etc.
        omega : float
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
    if np.allclose(xx[1,0], 0):
        start = 2
    else:
        start = 1
            
    return units.e**2 * (xx[0,start:].dot(xx[start:,0] * sos.D1(E, omega, units, start)) 
        + xx[0,start:].dot(xx[start:,0] * sos.D1(E.conjugate(), -omega, units, start)))
