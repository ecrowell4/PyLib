import numpy as np

import nlopy
from nlopy import math_utils
from nlopy.quantum_solvers import evolver_utils

def take_RK_step(Psi, Vfunc, t, dt):
    """Evolves psi(t) to psi(t+dt) via fourth order Runge-Kutta, using
    the Hartree Fock operator to evolve.

    Input
        Psi : class
            State of the system at time t
        V_func(x, t) : function
            function that returns potential at point x and time t

    Output
        Psi_new : class
            state of system at time t+dt
    """
    # Determine grid spacing
    dx = Psi.x[1] - Psi.x[0]

    k1u = np.zeros((Psi.Nu, Psi.Nx)) + 0j
    k2u = np.zeros((Psi.Nu, Psi.Nx)) + 0j
    k3u = np.zeros((Psi.Nu, Psi.Nx)) + 0j
    k4u = np.zeros((Psi.Nu, Psi.Nx)) + 0j

    if Psi.Nd != 0:
        k1d = np.zeros((Psi.Nd, Psi.Nx)) + 0j
        k2d = np.zeros((Psi.Nd, Psi.Nx)) + 0j
        k3d = np.zeros((Psi.Nd, Psi.Nx)) + 0j
        k4d = np.zeros((Psi.Nd, Psi.Nx)) + 0j

    newPsi = Psi.get_copy()

    k1u = (-1j/Psi.hbar) * evolver_utils.apply_F(Psi, Vfunc(Psi.x, t), 'u')
    if Psi.Nd != 0:
        k1d = (-1j/Psi.hbar) * evolver_utils.apply_F(Psi, Vfunc(Psi.x, t), 'd')
    
    tmpPsi = Psi.get_copy()
    tmpPsi.psiu += dt * k1u / 2
    if Psi.Nd != 0:
        tmpPsi.psid += dt * k1d / 2
    k2u = (-1j/Psi.hbar) * evolver_utils.apply_F(tmpPsi, Vfunc(Psi.x, t+dt/2), 'u')
    if Psi.Nd != 0:
        k2d = (-1j/Psi.hbar) * evolver_utils.apply_F(tmpPsi, Vfunc(Psi.x, t+dt/2), 'd')
    
    tmpPsi = Psi.get_copy()
    tmpPsi.psiu += dt * k2u / 2
    if Psi.Nd != 0:
        tmpPsi.psid += dt * k2d / 2
    k3u = (-1j/Psi.hbar) * evolver_utils.apply_F(tmpPsi, Vfunc(Psi.x, t+dt/2), 'u')
    if Psi.Nd != 0:
        k3d = (-1j/Psi.hbar) * evolver_utils.apply_F(tmpPsi, Vfunc(Psi.x, t+dt/2), 'd')
    
    tmpPsi = Psi.get_copy()
    tmpPsi.psiu += dt * k3u
    if Psi.Nd != 0:
        tmpPsi.psid += dt * k3d
    k4u = (-1j/Psi.hbar) * evolver_utils.apply_F(tmpPsi, Vfunc(Psi.x, t+dt), 'u')
    if Psi.Nd != 0:
        k4d = (-1j/Psi.hbar) * evolver_utils.apply_F(tmpPsi, Vfunc(Psi.x, t+dt), 'd')
 
    # Take time step
    newPsi.psiu = newPsi.psiu + (dt/6) * (k1u + 2*k2u + 2*k3u + k4u)
    if Psi.Nd != 0:
        newPsi.psid = newPsi.psid + (dt/6) * (k1d + 2*k2d + 2*k3d + k4d)
    
    return newPsi

def get_HF_energy(Psi, Vx):
    """Compute the total energy from the State Psi.

    Input
        Psi : class
            state of the system
        Vx : np.array
            external potential at time t

    Output
        E : float
            total energy
    """
    
    E = 0 + 0j

    Hpsiu = evolver_utils.apply_H(Psi.psiu, Psi.x, Vx, Psi.hbar, Psi.m, Psi.e)
    for a in range(Psi.Nu):
        E += math_utils.braket(Psi.psiu[a], Hpsiu[a], Psi.x)

    Jpsiu = evolver_utils.direct_integrals(Psi.psiu, Psi.Uc, Psi.x, Psi.e)
    Kpsiu = evolver_utils.exchange_integrals(Psi.psiu, Psi.Uc, Psi.x, Psi.e)
    for a in range(Psi.Nu):
        E += 0.5 * math_utils.braket(Psi.psiu[a], (Jpsiu[a] - Kpsiu[a]), Psi.x)

    if Psi.Nd != 0:
        Hpsid = evolver_utils.apply_H(Psi.psid, Psi.x, Vx, Psi.hbar, Psi.m, Psi.e)
        Jpsid = evolver_utils.direct_integrals(Psi.psid, Psi.Uc, Psi.x, Psi.e)
        Kpsid = evolver_utils.exchange_integrals(Psi.psid, Psi.Uc, Psi.x, Psi.e)
        for a in range(Psi.Nd):
            E += 0.5 * math_utils.braket(Psi.psid[a], (Hpsid[a] + Jpsid[a] - Kpsid[a]), Psi.x)

        rhou = np.sum(Psi.psiu.conjugate() * Psi.psiu, axis=0)
        Ucud = Psi.e**2 * math_utils.coulomb_convolve(rhou, Psi.Uc, Psi.x)
        for a in range(Psi.Nd):
            E +=  math_utils.braket(Psi.psid[a],  Psi.psid[a] * Ucud, Psi.x)

    return E