import numpy as np

import nlopy
from nlopy import math_utils
from nlopy.quantum_solvers import evolver_utils

def take_RK_step(Psi, Vfunc, t, dt, Psi_grnd=None):
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

    k1u = (-1j/Psi.hbar) * evolver_utils.apply_F(Psi, Vfunc(Psi.x, t), 'u', Psi.state, Psi_grnd)
    if Psi.Nd != 0:
        k1d = (-1j/Psi.hbar) * evolver_utils.apply_F(Psi, Vfunc(Psi.x, t), 'd', Psi.state, Psi_grnd)
    
    tmpPsi = Psi.get_copy()
    tmpPsi.psiu += dt * k1u / 2
    if Psi.Nd != 0:
        tmpPsi.psid += dt * k1d / 2
    k2u = (-1j/Psi.hbar) * evolver_utils.apply_F(tmpPsi, Vfunc(Psi.x, t+dt/2), 'u', Psi.state, Psi_grnd)
    if Psi.Nd != 0:
        k2d = (-1j/Psi.hbar) * evolver_utils.apply_F(tmpPsi, Vfunc(Psi.x, t+dt/2), 'd', Psi.state, Psi_grnd)
    
    tmpPsi = Psi.get_copy()
    tmpPsi.psiu += dt * k2u / 2
    if Psi.Nd != 0:
        tmpPsi.psid += dt * k2d / 2
    k3u = (-1j/Psi.hbar) * evolver_utils.apply_F(tmpPsi, Vfunc(Psi.x, t+dt/2), 'u', Psi.state, Psi_grnd)
    if Psi.Nd != 0:
        k3d = (-1j/Psi.hbar) * evolver_utils.apply_F(tmpPsi, Vfunc(Psi.x, t+dt/2), 'd', Psi.state, Psi_grnd)
    
    tmpPsi = Psi.get_copy()
    tmpPsi.psiu += dt * k3u
    if Psi.Nd != 0:
        tmpPsi.psid += dt * k3d
    k4u = (-1j/Psi.hbar) * evolver_utils.apply_F(tmpPsi, Vfunc(Psi.x, t+dt), 'u', Psi.state, Psi_grnd)
    if Psi.Nd != 0:
        k4d = (-1j/Psi.hbar) * evolver_utils.apply_F(tmpPsi, Vfunc(Psi.x, t+dt), 'd', Psi.state, Psi_grnd)
 
    # Take time step
    newPsi.psiu = newPsi.psiu + (dt/6) * (k1u + 2*k2u + 2*k3u + k4u)
    if Psi.Nd != 0:
        newPsi.psid = newPsi.psid + (dt/6) * (k1d + 2*k2d + 2*k3d + k4d)
    
    return newPsi

def get_orbital_energies(Psi, Vx):
    """Compute the orbital energies.

    Input
        Psi : class
            state object
        Vx : np.array
            external potential

    Output
        Esu, Esd : np.array
            orbital energies of the spin up and down orbitals, respectively
    """
    tmpPsi = Psi.get_copy()
    tmpPsi.lagrange = False
    Esu = np.zeros((Psi.Nu, Psi.Nu)) + 0j
    Esd = np.zeros((Psi.Nd, Psi.Nd)) + 0j
    Fpsiu = evolver_utils.apply_F(tmpPsi, Vx, 'u')
    Fpsid = evolver_utils.apply_F(tmpPsi, Vx, 'd')
    for i in range(tmpPsi.Nu):
        for j in range(tmpPsi.Nd):
            Esu[i, j] = math_utils.braket(tmpPsi.psiu[i], Fpsiu[j], tmpPsi.x)
    for i in range(tmpPsi.Nd):
        for j in range(tmpPsi.Nd):
            Esd[i, j] = math_utils.braket(tmpPsi.psid[i], Fpsid[j], tmpPsi.x)
    return Esu, Esd

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

    Jpsiuu = evolver_utils.direct_integrals(Psi.psiu, Psi.psiu, Psi.Uc, Psi.x, Psi.e)
    Kpsiu = evolver_utils.exchange_integrals(Psi.psiu, Psi.Uc, Psi.x, Psi.e)
    for a in range(Psi.Nu):
        E += 0.5 * math_utils.braket(Psi.psiu[a], (Jpsiuu[a] - Kpsiu[a]), Psi.x)

    if Psi.Nd != 0:
        Hpsid = evolver_utils.apply_H(Psi.psid, Psi.x, Vx, Psi.hbar, Psi.m, Psi.e)
        Jpsidd = evolver_utils.direct_integrals(Psi.psid, Psi.psid, Psi.Uc, Psi.x, Psi.e)
        Kpsid = evolver_utils.exchange_integrals(Psi.psid, Psi.Uc, Psi.x, Psi.e)
        for a in range(Psi.Nd):
        	E += math_utils.braket(Psi.psid[a], Hpsid[a], Psi.x)
        for a in range(Psi.Nd):
            E += 0.5 * math_utils.braket(Psi.psid[a], (Jpsidd[a] - Kpsid[a]), Psi.x)

        Jpsiud = evolver_utils.direct_integrals(Psi.psid, Psi.psiu, Psi.Uc, Psi.x, Psi.e)
        for a in range(Psi.Nu):
            E +=  math_utils.braket(Psi.psiu[a], Jpsiud[a], Psi.x)

    return E