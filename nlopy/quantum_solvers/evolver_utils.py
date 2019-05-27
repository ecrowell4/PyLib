
import numpy as np
from numba import jit
import scipy
from scipy import integrate

import nlopy
from nlopy import utils
from nlopy.quantum_solvers import solver_1D, solver_utils

def apply_F(Psi, Vx, spin):
	"""Returns the action of HF operator on orbitals for paritcular
	spin state.

	Input
	    Psi : class
	        state of system
	    Vx : np.array
	        external potential
	    spin : string
	        'u' or 'd': string indicating spin of electron. 

	Output
	    Fpsi : the action of HF operator on orbitals of given spin.
	"""

    if spin is 'u':
    	psia = Psi.psiu
    	psib = Psi.psid
    if spin is 'd':
    	psia = Psi.psid
    	psib = Psi.psiu
    Hpsia = apply_H(psia, x, Vx, Psi.hbar, Psi.m, Psi.e)
	Jpsia = direct_integrals(psia, x, q)
	Jpsib = direct_integrals(psib, x, q)
	Kpsia = exchange_integrals(psia, x, q)
	return Hpsia + Jpsia - Kpsia + Jpsib

