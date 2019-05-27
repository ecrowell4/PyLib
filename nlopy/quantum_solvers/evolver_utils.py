
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
        Hpsi[n] = -(hbar**2/2/m) * math_utils.laplacian(psi[n], dx) + Vx*psi[n]
    return Hpsi

@jit(nopython=True)
def direct_integrals(psi:complex, x:float, q:float)->complex:
    """Returns the direct integrals for orbitals of given type.
    Note that the direct integral for all of the orbitals is the same.
    This allows us to only compute a single integral, then broadcast
    the result.

    Input
        psi : np.array
            psi[i] is the ith orbital of specific spin type
        x : np.array
            spatial array
        q : float
            charge of particle

    Output
        Jpsi : np.array
            Jpsi[i] is action of the Coulomb operator on orbital psi[i]
    """

    Norb:int = len(psi)
    rho:float = np.sum(psi.conjugate() * psi, axis=0)
    coulomb_operator:complex = my_convolve(rho, x)
    J:complex = np.zeros(psi.shape) + 0j
    for n in range(Norb): 
        J[n] = coulomb_operator * psi[n]
    return J