import numpy as np
from nlopy.hyperpol.sum_over_states import sos_utils

def magnetic_moment_3rd(E, L, I, omega, Efield, Bfield, units):
    
    H1 = -units.e * xx * Efield - units.g * L * Bfield 
    H2 = -(units.e**2 / 2 / units.m / omega[0] / omega[1]) Efield**2
    
    