
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

@jit(nopython=True)
def apply_H(psi:complex, x:float, Vx:float, hbar:float, m:float, e:float)->complex:
	"""Returns the action of single particle hamiltonian on
	set of orbitals.

	Input
	    psi : np.array
	        collection of orbitals of same spin type.
	        psi[i] is ith orbital
	    x : np.array
	        spatial array
	    Vx : np.array
	        external potential
	    hbar, m, e : float
	        planck'c constant, particle mass, particle charge.
	        All equal to 1 in atomic units.

	Output
	    Hpsi : np.array
	        Hpsi[i] is action of H on ith orbital psi[i]
	"""
    
    dx:float = x[1] - x[0]
    Hpsi:complex = np.zeros(psi.shape) + 0j
    Norb:int = len(psi)
    for n in range(Norb):
    	Hpsi[n] = -(hbar**2/2/m) * laplacian(psi[n], dx) + Vx*psi[n]
    return Hpsi
