import numpy as np
from nlopy.quantum_solvers import solver_utils
from nlopy.quantum_solvers import many_electron_utils
from concurrent import futures
import functools

def take_step_split_op(psi, V_func, x, t, dt, units):
    """Evolves psi(t) to psi(t+dt) via the split operator method.

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
    

def take_step_RungeKutta(psi, V_func, x, t, dt, units):
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

    # Take step
    psi = psi + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    # normalize
    psi = psi / np.sqrt(np.trapz(abs(psi)**2, x))

def take_step_RungeKutta_HF(psi, V_func, Ne, x, t, dt, units):
    """Evolves psi(t) to psi(t+dt) via fourth order Runge-Kutta, using
    the Hartree Fock operator to evolve.

    Input
        psi : np.array
            state vectors at time t. psi[i] is the ith particle state.
        V_func(x, t) : function
            function that returns potential at point x and time t
        a : np.int
            state we are evolving in presence of other electrons
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

    for a in range(Ne):
        # Compute Runge-Kutta coefficients
        k1 = (-1j / units.hbar) * many_electron_utils.apply_f(x, psi[a], psi, V_func(x, t), a, Ne, units)
        k2 = (-1j / units.hbar) * many_electron_utils.apply_f(x, psi[a] + (dt * k1 / 2), psi, V_func(x, t + dt / 2), a, Ne, units)
        k3 = (-1j / units.hbar) * many_electron_utils.apply_f(x, psi[a] + (dt * k2 / 2), psi, V_func(x, t + dt / 2), a, Ne, units)
        k4 = (-1j / units.hbar) * many_electron_utils.apply_f(x, psi[a] + (dt * k3), psi, V_func(x, t + dt), a, Ne, units)

        psi[a] = psi[a] + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        psi[a] = psi[a] / np.sqrt(np.trapz(abs(psi[a])**2, x))
    
    return psi

def take_parallel_step(a, psi, V_func, Ne, x, t, dt, units):
    """Evolves psi(t) to psi(t+dt) via fourth order Runge-Kutta, using
    the Hartree Fock operator to evolve.

    Input
        a : np.int
            particle state to be evolved
        psi : np.array
            state vectors at time t. psi[i] is the ith particle state.
        V_func(x, t) : function
            function that returns potential at point x and time t
        a : np.int
            state we are evolving in presence of other electrons
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

    k1 = (-1j / units.hbar) * many_electron_utils.apply_f(x, psi[a], psi, V_func(x, t), a, Ne, units)
    k2 = (-1j / units.hbar) * many_electron_utils.apply_f(x, psi[a] + (dt * k1 / 2), psi, V_func(x, t + dt / 2), a, Ne, units)
    k3 = (-1j / units.hbar) * many_electron_utils.apply_f(x, psi[a] + (dt * k2 / 2), psi, V_func(x, t + dt / 2), a, Ne, units)
    k4 = (-1j / units.hbar) * many_electron_utils.apply_f(x, psi[a] + (dt * k3), psi, V_func(x, t + dt), a, Ne, units)

    psi[a] = psi[a] + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    psi[a] = psi[a] / np.sqrt(np.trapz(abs(psi[a])**2, x))
    
    return psi[a] 

def take_step_RungeKutta_HF_parallel(psi, V_func, Ne, x, t, dt, units):
    """Evolves psi(t) to psi(t+dt) via fourth order Runge-Kutta, using
    the Hartree Fock operator to evolve.

    Input
        a : np.int
            particle state to be evolved
        psi : np.array
            state vectors at time t. psi[i] is the ith particle state.
        V_func(x, t) : function
            function that returns potential at point x and time t
        a : np.int
            state we are evolving in presence of other electrons
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
    
    ppool = futures.ThreadPoolExecutor(20)
    psi = np.asarray( list( ppool.map( 
            functools.partial(take_parallel_step, psi=psi, V_func=V_func, Ne=Ne, x=x, t=t, dt=dt, units=units )
            , np.arange(Ne))
    ), dtype=object)
    
    return psi
    
    

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
        psis[counter+1] = take_step_RungeKutta(psis[counter], V_func, x, t, dt, units)

    return psis

