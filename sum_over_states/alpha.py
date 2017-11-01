# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 12:38:46 2017

@author: Ethan Crowell
"""

import numpy as np

import sys
sys.path.append(r'C:\Users\Owner\Documents\PyLib')

from PyLib.sum_over_states import sos_utils as sos

    
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
    return units.e**2 * (xx[0,1:].dot(xx[1:,0] * sos.D1(E, omega)) 
        + xx[0,1:].dot(xx[1:,0] * sos.D1(E.conjugate(), -omega)))
