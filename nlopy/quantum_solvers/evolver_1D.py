import numpy as np
from nlopy.quantum_solvers import solver_utils

def take_time_step(psi, V_func, x, t, dt, units):
    """Evolves psi(t) to psi(t+dt) via fourth order Runge-Kutta.

    Input
        psi : np.array
            state vector at time t
        V_func(x, t) : function
            function that returns potential at point x and time t
        x : np.array
            spatial array
        t : float
            current time
        dt : float
            time step size
        units : Class
            object containing fundamental constants

    Output
        psi : np.array
            state vector at time t+dt
    """

    # Compute Runge-Kutta coefficients
    k1 = (-1j / units.hbar) * solver_utils.apply_H(psi, x, V_func(x, t), units)
    k2 = (-1j / units.hbar) * solver_utils.apply_H(psi + (dt * k1 / 2), x, V_func(x, t + dt / 2), units)
    k3 = (-1j / units.hbar) * solver_utils.apply_H(psi + (dt * k2 / 2), x, V_func(x, t + dt / 2), units)
    k4 = (-1j / units.hbar) * solver_utils.apply_H(psi + (dt * k3), x, V_func(x, t + dt), units)

    psi = psi + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    return psi / np.sqrt(np.trapz(abs(psi)**2, x))


def evolve(psi0, V_func, x, T, units):
    """Evolves the state psi0 over the time doma
    in T.

    Input
        psi0 : np.array
            initial state
        V_func(x, t) : function
            function that returns the potential at point x and time t
        x, T : np.array
            spatial and temporal array
        units : Class
            object containing fundamental constants

    Output
        psis : np.array
            psis[i] is state vector at ith time step
    """

    # Determine cardinality of space and time arrays
    Nt = len(T)
    Nx = len(x)
    dt = T[1] - T[0]

    # Create array to store state vectors
    psis = np.zeros((Nt, Nx), dtype=complex)
    psis[0] = psi0

    # Propogate in time
    for counter, t in enumerate(T[:-1]):
        psis[counter+1] = take_time_step(psis[counter], V_func, x, t, dt, units)

    return psis

