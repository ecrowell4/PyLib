import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import functools; import time
#from scipy.interpolate import griddata

import nlopy
from nlopy import utils, math_utils
from nlopy.quantum_solvers import solver_1D, solver_utils, evolver_1D, many_electron_utils
from nlopy.quantum_solvers import evolver_HF

units = utils.Units('atomic')
etol = 1e-8

def URHF(Psi0, Vfunc, dt, tol, max_iter=int(1e4)):
    """Uses a diffusive Hartree-Fock method (imaginary time) to 
    compute the lowest energy state.

    Input
        Psi0 : class
            initial state of the system
        Vfunc : function
            function which returns potential at given time and position

    Output
        Psif : class
            final state of system
        Es : np.array
            energie at each step in imaginary time.
    """

    if Psi0.state == 'excited':
        l = 0
        Psi_grnd = utils.load_class('../data_files/unrestricted_HF/states_'+str(Psi0.Nu)+'u_'+str(Psi0.Nd)+'d_el_2a0_ground_ohno.pkl')
        #Psi_grnd = utils.load_class('../data_files/unrestricted_HF/states_'+str(Psi0.Nu)+'u_'+str(Psi0.Nd)+'d_el_2a0_ground.pkl')
        #Psi_grnd = utils.load_class('../../nlopy_test/ground_state.pkl')
        for i in range(Psi0.Nu):
            Psi0.psiu[l] -= math_utils.project(Psi0.psiu[l], Psi_grnd.psiu[i], Psi0.x)
        Psi0.psiu[l] /= np.sqrt(math_utils.braket(Psi0.psiu[l], Psi0.psiu[l], Psi0.x))
    elif Psi0.state == 'ground':
    	Psi_grnd = None

    ediff = 1
    Es = np.zeros(0)
    Es = np.append(Es, evolver_HF.get_HF_energy(Psi0, Vfunc(Psi0.x, t=0)))
    tmpPsi = Psi0.get_copy()
    Psif = Psi0.get_copy()
    n=1
    while np.allclose(0, ediff, atol=tol) is False:
        #print(n)
        start = time.time()
        tmpPsi = evolver_HF.take_RK_step(Psif, Vfunc, -1j*n*dt, -1j*dt, Psi_grnd)
        if Psi0.state is 'excited':
            l = 0
            for i in range(tmpPsi.Nu):
                tmpPsi.psiu[l] -= math_utils.project(tmpPsi.psiu[l], Psi_grnd.psiu[i], tmpPsi.x)
                tmpPsi.psiu[l] /= np.sqrt(math_utils.braket(tmpPsi.psiu[l], tmpPsi.psiu[l], tmpPsi.x))
        if Psi0.Nu != 0:
            tmpPsi.psiu = math_utils.gram_schmidt(tmpPsi.psiu, tmpPsi.x)
        if Psi0.Nd != 0:
            tmpPsi.psid = math_utils.gram_schmidt(tmpPsi.psid, tmpPsi.x)
        tmpE = evolver_HF.get_HF_energy(tmpPsi, Vfunc(tmpPsi.x, -1j*n*dt))
        Psif = tmpPsi.get_copy()
        Es = np.append(Es, tmpE)
        ediff = (Es[n] - Es[n-1])/dt
        n +=1
        end = time.time()
        #print('Process took %.3f seconds' % (end-start))
        if n >= max_iter:
            print("Exceeded maximum number of iterations.")
            break
    Psif.Energy = Es[-1]
    return Psif, Es


def minimize_energy(psi0, x, V, Ne, units, lagrange=True, exchange=False, etol=1e-8, fft=False, exc_state=False, psi_grnd=None):
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
        exc_state : bool
            if True, get first excited state instead of ground state
        psi_grnd : np.array, None
            actual ground state. Only needed when finding excited state

    Output 
        psi : np.array
             array of single particle orbitals that minimize the
             energy.
        Es : np.array
            many electron energies for each iteration
    """

    if exc_state is True:
        assert psi_grnd is not None, "Err: Need ground state in order to get excited state."
    
    N_orb = int(Ne / 2)
    L = x[-1] - x[0]
    N = len(x)
    dx = x[1] - x[0]
    dt = dx**2 / 2
    psi = psi0

    if exc_state is True:
        # Project out the ground state from initial state
        for n in range(N_orb):
            psi[-1] = many_electron_utils.project_off(psi[-1], psi_grnd[n], x)
            assert np.allclose(many_electron_utils.braket_jit(psi[-1], psi_grnd[n], x), 0), "Not make_orthogonal" 

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
        if exc_state is True:
            # Project out the ground state from initial state
            for l in range(N_orb):
                psi_temp[-1] = many_electron_utils.project_off(psi_temp[-1], psi_grnd[l], x)
                assert np.allclose(many_electron_utils.braket_jit(psi_temp[-1], psi_grnd[l], x), 0), "Not make_orthogonal"    

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
        # elif E_temp > Es[n-1]:
        #     print('Going uphill')
        #     #print(E_temp)
        #     dt = many_electron_utils.update_dt(dt, 'decrease', delta=0.1)
        #     print(dt)
        else:
            psi = psi_temp
            Es = np.append(Es, E_temp)
            ediff_ = (Es[n] - Es[n-1]) / dt
            ediff = abs(Es[n] - Es[n-1]) / dt
            success_number += 1
            n += 1
            print('Energy difference = '+str(ediff_))
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

        if exc_state is True:
            # Project out the ground state from initial state
            for n in range(N_orb):
                psi[-1] = many_electron_utils.project_off(psi[-1], psi_grnd[n], x)

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
        # elif E_temp > Es[n-1]:
        #     #print('Going uphill')
        #     #print(E_temp)
        #     dt = many_electron_utils.update_dt(dt, 'decrease', delta=0.1)
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