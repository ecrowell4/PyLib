import numpy as np
import scipy.integrate as sp_integrate
import copy
import sys

def get_local_field_L1(N, alpha):
	"""Returns the first local field factor for a dielectric.

	Input
	    N : float
	        number density of units (per a0^3)
	    alpha : complex
	        polarizability (in atomic units)

	Output
	    L1 : complex
	        first local field factor
	"""
	return 3 / (3 - 4*np.pi*N*alpha)

	