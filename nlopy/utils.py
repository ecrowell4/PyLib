
"""This file containts funtions and classes that should be of general use in Nonlinear optics, regardless
of application."""

import numpy as np
from numba import jit

def position_space(L, N, centered=False, periodic=False):
    """Returns an array that represents a discretized representation of
    the interval [L_min, L_min + L], enpoints included. 

    Input
        L : float
            Length of position space
        N : int
            number of points
        centered : bool
            if True, origin is at center of domain. Otherwise, origin 
            is at left.
        periodic : bool
            True if domain is periodic, in which case endpoint is excluded
            to avoid redundancy.

    Output
        x : np.array
            Spatial grid
    """

    dx = L / (N-1)

    if centered is True:
        left_adjust = L / 2
    else:
        left_adjust = 0

    if periodic is True:
        x = np.arange(N - 1) * dx - left_adjust
    else:
        x = np.arange(N) * dx - left_adjust

    return x
    
class Units():
    """Class whose attributes are the fundamental constants. User can choose from
    ['atomic', 'Gaussian'].
    Input
        unit_type : string
            Specifies the desired type of units
    """

    def __init__(self, unit_type):
        if unit_type is 'atomic':
            self.hbar = 1
            self.e = 1
            self.m= 1
            self.c = 137.036
            self.g = self.e / 2 / self.m / self.c  #gyromagnetic ratio
        elif unit_type is 'Gaussian':
            self.hbar = 6.58211928e-16 #eV.s
            self.e = 4.8032042e-10 #esu
            self.c = 2.99792458e10 #cm/s
            self.m = 0.51099890e6 / (self.c)**2 #eV.s^2/cm^2 
            self.g = self.e / 2 / self.m / self.c  #gyromagnetic ratio
        else:
            assert False, "Indicated units not supported. Please choose from ['atomic', Gaussian']."
            

