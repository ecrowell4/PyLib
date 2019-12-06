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

def get_chi1(N, alpha, orient='random'):
	"""Returns the first electric susceptibility, including
	the orientational average.

	Input
	    N : float
	        number density (per a0^3)
	    alpha : complex
	        polarizability (in atomic units)
	    orient : string
	        specifies orientation of atoms. Code only 
	        implements random orientation for now.

	Output
	    chi1 : complex
	        first susceptibility
	"""
	L1 = get_local_field_L1(N, alpha)
	return (2/3) * N * L1 * alpha