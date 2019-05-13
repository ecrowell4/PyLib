import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import functools; import time
#from scipy.interpolate import griddata

import nlopy
from nlopy import utils
from nlopy.quantum_solvers import solver_1D, solver_utils, evolver_1D, many_electron_utils

units = utils.Units('atomic')
etol = 1e-8

def minimize_energy(psi0, V, Ne, units, lagrange=True, exchange=False, etol=1e-8, fft=False):
    """Returns the single particle orbitals whose direct product (or slater det
    if exchange is True) minimizes the many electron energy.

    Input
        Ne : int
            number of electrons. Must be even for closed shell!
        V : function
            external potential. V(x, t) return potential
            at position x and time t
        units : class
            fundamnetal constants
        lagrange : bool
            if true, use lagrange multipliers to keep states
            orthonormal
        exchange : bool
            if true, use Hartree Fock instead of Hartree
        etol : float
            tolerance for derivatives convergence.
        fft : bool
            if true, use fourier methods for derivatives

    Output 
        psi : np.array
             array of single particle orbitals that minimize the
             energy.
        Es : np.array
            many electron energies for each iteration
    """
    
    N_orb = int(Ne / 2)
    L = 2 * Ne
    N = 20 * Ne + 1
    dx = L / (N-1)
    dt = dx**2 / 2
    x = utils.position_space(L, N)
    psi = psi0

    # Create array to store energies at each iteration
    Es = np.zeros(0, dtype=complex)
    Es = np.append(Es, many_electron_utils.get_HF_energy(x, psi0, V(x, 0), N_orb, units))

    n = 1
    ediff = 1
    success_number = 0
    while np.allclose(ediff, 0, atol=etol) is False:
        
        start = time.time()
        # Get states at next time step
        psi_temp = evolver_1D.take_step_RungeKutta_HF(psi, V, N_orb, x, 
                                                 -1j*n*dt, -1j*dt, units, lagrange=lagrange,
                                                  exchange=exchange, fft=fft)        

        # Renormalize/Reorthogonalize
        if lagrange is False:
            psi_temp = many_electron_utils.make_orthogonal(psi_temp, dx)
        elif lagrange is True:
            psi_temp = many_electron_utils.gram_schmidt_jit(psi_temp, x)


        # Compute energy of new configuration
        E_temp = many_electron_utils.get_HF_energy(x, psi_temp.astype(complex), 
            V(x, -1j*n*dt), N_orb, units, exchange=exchange, fft=fft)    

        end = time.time()
        #print("step "+str(n)+" took %.3f seconds" % (end - start))

        # Compute overlap matrix of new configuration
        S = many_electron_utils.overlap_matrix_jit(psi_temp, x)
        
        # Ensure that we're going downhill and that states are orthonormal

        if np.allclose(S, np.eye(N_orb), atol=1e-2)==False:
            success_number = 0
            print("States not orthonormal.")
            dt = many_electron_utils.update_dt(dt, 'decrease', delta=0.1)
        elif E_temp > Es[n-1]:
            print('Going uphill')
            #print(E_temp)
            dt = many_electron_utils.update_dt(dt, 'decrease', delta=0.1)
        else:
            psi = psi_temp
            Es = np.append(Es, E_temp)
            ediff = abs(Es[n] - Es[n-1]) / dt
            success_number += 1
            n += 1
            #print('Energy difference = '+str(ediff))
            if success_number % 10 == 0:
                success_number = 0
                dt_tmp = many_electron_utils.update_dt(dt, 'increase', delta=0.1)
                if dt_tmp < dx**2 / 2:
                    dt = dt_tmp
    time.sleep(2)
    for i in range(100):
        #print(i)
        start = time.time()
        # Get states at next time step
        psi_temp = evolver_1D.take_step_RungeKutta_HF(psi, V, N_orb, x, 
                                                 -1j*n*dt, -1j*dt, units, lagrange=lagrange,
                                                  exchange=exchange, fft=fft)        

        # Renormalize/Reorthogonalize
        if lagrange is False:
            psi_temp = many_electron_utils.make_orthogonal(psi_temp, dx)
        elif lagrange is True:
            psi_temp = many_electron_utils.gram_schmidt_jit(psi_temp, x)


        # Compute energy of new configuration
        E_temp = many_electron_utils.get_HF_energy(x, psi_temp.astype(complex), 
            V(x, -1j*n*dt), N_orb, units, exchange=exchange, fft=fft)    

        end = time.time()
        #print("step "+str(n)+" took %.3f seconds" % (end - start))

        # Compute overlap matrix of new configuration
        S = many_electron_utils.overlap_matrix_jit(psi_temp, x)
        
        # Ensure that we're going downhill and that states are orthonormal

        if np.allclose(S, np.eye(N_orb), atol=1e-2)==False:
            success_number = 0
            print("States not orthonormal.")
            dt = many_electron_utils.update_dt(dt, 'decrease', delta=0.1)
        elif E_temp > Es[n-1]:
            #print('Going uphill')
            #print(E_temp)
            dt = many_electron_utils.update_dt(dt, 'decrease', delta=0.1)
        else:
            psi = psi_temp
            Es = np.append(Es, E_temp)
            ediff = abs(Es[n] - Es[n-1]) / dt
            success_number += 1
            n += 1
            #print('Energy difference = '+str(ediff))
            if success_number % 10 == 0:
                success_number = 0
                dt_tmp = many_electron_utils.update_dt(dt, 'increase', delta=0.1)
                if dt_tmp < dx**2 / 2:
                    dt = dt_tmp
    return psi, Es