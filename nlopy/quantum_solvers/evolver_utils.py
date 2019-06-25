
import numpy as np
from numba import jit
import scipy
from scipy import integrate

import nlopy
from nlopy import utils, math_utils
from nlopy.quantum_solvers import solver_1D, solver_utils

def apply_F(Psi, Vx, spin, state='ground', Psi_grnd=None):
    """Returns the action of HF operator on orbitals for paritcular
    spin state.

    Input
        Psi : class
            state of system
        Vx : np.array
            external potential
        spin : string
            'u' or 'd': string indicating spin of electron. 
        state : string
            'ground' or 'excited'

    Output
        Fpsi : array
            the action of HF operator on orbitals of given spin.
    """
    if state is 'excited':
        assert Psi_grnd is not None, "Must have ground state information to find excited states."
    if spin is 'u':
        psia = Psi.psiu
        psib = Psi.psid
    if spin is 'd':
        psia = Psi.psid
        psib = Psi.psiu
    Hpsia = apply_H(psia, Psi.x, Vx, Psi.hbar, Psi.m, Psi.e)
    Jpsia = direct_integrals(psia, psia, Psi.Uc, Psi.x, Psi.e)
    if len(psib) != 0:
        Jpsib = direct_integrals(psib, psia, Psi.Uc, Psi.x, Psi.e)
    else:
        Jpsib = np.zeros(Jpsia.shape)
        
    Kpsia = exchange_integrals(psia, Psi.Uc, Psi.x, Psi.e)
    Fpsia = Hpsia + Jpsia + Jpsib - Kpsia
    if Psi.lagrange is True:
        Fpsia = math_utils.subtract_lagrange(Fpsia, psia, Psi.x)
    if state is 'excited' and spin is 'up':
        for i in range(Psi_grnd.Nu):
            Fpsia[-1] -= math_utils.project(Fpsia[-1], Psi_grnd.psiu[i], Psi.x)
    return Fpsia

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
def direct_integrals(psi1:complex, psi2:complex, Uc:float, x:float, q:float)->complex:
    """Returns the direct integrals for orbitals of given type.
    Note that the direct integral for all of the orbitals is the same.
    This allows us to only compute a single integral, then broadcast
    the result.

    Input
        psi1,2 : np.array
            psi[i] is the ith orbital of specific spin type.
        Uc : np.array
            Coulomb kernel in position space (i.e. 1/r, |r|, log(r), etc.)
        x : np.array
            spatial array
        q : float
            charge of particle

    Output
        Jpsi : np.array
            Jpsi[i] is action of the Coulomb operator on orbital psi[i]
    """

    Norb:int = len(psi2)
    rho1:float = np.sum(psi1.conjugate() * psi1, axis=0)
    coulomb_operator:complex = q**2 * math_utils.coulomb_convolve(rho1, Uc, x)
    J:complex = np.zeros(psi2.shape) + 0j
    for n in range(Norb): 
        J[n] = coulomb_operator * psi2[n]
    return J

@jit(nopython=True)
def exchange_integrals(psi:complex, Uc:float, x:float, q:float)->complex:
    """Returns action of exchange operator on each orbital of given
    type.

    Input
        psi : np.array
            psi[i] is ith orbital
        Uc : np.array
            Coulomb kernel in position space
        x : np.array
            spatial array
        q : float
            charge of particle
    Output 
        Kpsi : np.array
            Kpsi[i] is action of exchange operator on orbital psi[i]
    """

    Norb:int = len(psi)
    K:complex = np.zeros(psi.shape) + 0j
    for i in range(Norb):
        for j in range(Norb):
            K[i] = K[i] + q**2 * math_utils.coulomb_convolve(psi[j].conjugate()*psi[i], Uc, x) * psi[j]
    return K

