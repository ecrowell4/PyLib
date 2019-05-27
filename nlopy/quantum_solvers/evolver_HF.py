import numpy as np
from nlopy.quantum_solvers import solver_utils
from nlopy.quantum_solvers import evolver_utils
from concurrent import futures
import functools

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
    dx = x[1] - x[0]

    k1u = np.zeros((State.Nu, State.Nx)) + 0j
    k2u = np.zeros((State.Nu, State.Nx)) + 0j
    k3u = np.zeros((State.Nu, State.Nx)) + 0j
    k4u = np.zeros((State.Nu, State.Nx)) + 0j

    k1d = np.zeros((State.Nd, State.Nx)) + 0j
    k2d = np.zeros((State.Nd, State.Nx)) + 0j
    k3d = np.zeros((State.Nd, State.Nx)) + 0j
    k4d = np.zeros((State.Nd, State.Nx)) + 0j

    newPsi = Psi.get_copy()
    tmpPsi = Psi.get_copy()

    k1u = evolver_utils.apply_F(Psi, Vfunc(Psi.x, t), 'u')
    k1u = evolver_utils.apply_F(Psi, Vfunc(Psi.x, t), 'd')
    tmpPsi.psiu = Psi.psiu + dt * k1u / 2
    tmpPsi.psid = Psi.psiu + dt * k1d / 2

    k2u = evolver_utils.apply_F(tmpPsi, Vfunc(Psi.x, t+dt/2), 'u')
    k2u = evolver_utils.apply_F(tmpPsi, Vfunc(Psi.x, t+dt/2), 'd')
    tmpPsi.psiu = Psi.psiu + dt * k2u / 2
    tmpPsi.psid = Psi.psiu + dt * k2d / 2

    k3u = evolver_utils.apply_F(tmpPsi, Vfunc(Psi.x, t+dt/2), 'u')
    k3u = evolver_utils.apply_F(tmpPsi, Vfunc(Psi.x, t+dt/2), 'd')
    tmpPsi.psiu = Psi.psiu + dt * k3u
    tmpPsi.psid = Psi.psiu + dt * k3d

    k4u = evolver_utils.apply_F(tmpPsi, Vfunc(Psi.x, t+dt), 'u')
    k4u = evolver_utils.apply_F(tmpPsi, Vfunc(Psi.x, t+dt), 'd')
 
    # Take time step
    newPsi.psiu = Psi.psiu + (dt/6) * (k1u + 2*k2u + 2*k3u + k4u)
    newPsi.psid = Psi.psid + (dt/6) * (k1d + 2*k2d + 2*k3d + k4d)
    
    return newPsi

    
