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
    k1 = (-1j / units.hbar) * np.append(np.append(0, solver_utils.apply_H(psi, x, V_func(x, t), units)[1:-1]),0)
    k2 = (-1j / units.hbar) * np.append(np.append(0, solver_utils.apply_H(psi + (dt * k1 / 2), x, V_func(x, t + dt / 2), units)[1:-1]), 0)
    k3 = (-1j / units.hbar) * np.append(np.append(0, solver_utils.apply_H(psi + (dt * k2 / 2), x, V_func(x, t + dt / 2), units)[1:-1]), 0)
    k4 = (-1j / units.hbar) * np.append(np.append(0, solver_utils.apply_H(psi + (dt * k3), x, V_func(x, t + dt), units)[1:-1]), 0)

    # Take step
    psi = psi + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    # normalize
    psi = psi / np.sqrt(np.trapz(abs(psi)**2, x))
    
    return psi

def take_step_RungeKutta_HF(psi, V_func, Ne, x, t, dt, units, lagrange):
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
    
    # Determine grid spacing
    dx = x[1] - x[0]
    
    psi_next = np.zeros(psi.shape, dtype=complex)

    for a in range(Ne):
        # Compute Runge-Kutta coefficients
        k1 = (-1j / units.hbar) * np.append(np.append(0, many_electron_utils.apply_f(x, psi[a], psi, 
             V_func(x, t), a, Ne, units, lagrange=lagrange)[1:-1]), 0)
        k2 = (-1j / units.hbar) * np.append(np.append(0, many_electron_utils.apply_f(x, psi[a] + (dt * k1 / 2), psi, 
              V_func(x, t + dt / 2), a, Ne, units, lagrange=lagrange)[1:-1]), 0)
        k3 = (-1j / units.hbar) * np.append(np.append(0, many_electron_utils.apply_f(x, psi[a] + (dt * k2 / 2), psi, 
              V_func(x, t + dt / 2), a, Ne, units, lagrange=lagrange)[1:-1]), 0)
        k4 = (-1j / units.hbar) * np.append(np.append(0, many_electron_utils.apply_f(x, psi[a] + (dt * k3), psi, 
              V_func(x, t + dt), a, Ne, units, lagrange=lagrange)[1:-1]), 0)

        psi_next[a][1:-1] = psi[a][1:-1] + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)[1:-1]
        psi_next[a] = psi_next[a] / np.sqrt(np.trapz(abs(psi_next[a])**2, dx=dx))
    
    return psi_next

def take_parallel_step(a, psi, V_func, Ne, x, t, dt, units, lagrange):
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
    # Determine grid spacing
    dx = x[1] - x[0]

    k1 = (-1j / units.hbar) * np.append(np.append(0, many_electron_utils.apply_f(x, psi[a], psi, 
         V_func(x, t), a, Ne, units, lagrange=lagrange)[1:-1]), 0)
    k2 = (-1j / units.hbar) * np.append(np.append(0, many_electron_utils.apply_f(x, psi[a] + (dt * k1 / 2), psi, 
          V_func(x, t + dt / 2), a, Ne, units, lagrange=lagrange)[1:-1]), 0)
    k3 = (-1j / units.hbar) * np.append(np.append(0, many_electron_utils.apply_f(x, psi[a] + (dt * k2 / 2), psi, 
          V_func(x, t + dt / 2), a, Ne, units, lagrange=lagrange)[1:-1]), 0)
    k4 = (-1j / units.hbar) * np.append(np.append(0, many_electron_utils.apply_f(x, psi[a] + (dt * k3), psi, 
          V_func(x, t + dt), a, Ne, units, lagrange=lagrange)[1:-1]), 0)

    psi[a] = psi[a] + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    psi[a] = psi[a] / np.sqrt(np.trapz(abs(psi[a])**2, dx=dx))
    
    return psi[a] 

def take_step_RungeKutta_HF_(psi, V_func, Ne, x, t, dt, units, lagrange=True):
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
    
    ppool = futures.ThreadPoolExecutor(4)
    psi = np.asarray( list( ppool.map( 
            functools.partial(take_parallel_step, psi=psi, V_func=V_func, Ne=Ne, x=x, t=t, dt=dt, units=units, lagrange=lagrange)
            , np.arange(Ne))
    ), dtype=complex)
    
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

