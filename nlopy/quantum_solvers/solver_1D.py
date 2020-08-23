import scipy as sp
from scipy import linalg
import scipy.linalg
import numpy as np
from nlopy.quantum_solvers import solver_utils

def solver_1D(x, V, units, num_states=15):
    """Returns eigenstates and eigenenergies of one dimensional potentials
    define on grid using a second order finite difference approximation
    for the kinetic energy. 
    
    Assumes infinite walls at both ends of the problem space.
    
    Input
        x : np.array([x_i]) 
            The spacial grid points including the endpoints
        V : np.array([V(x_i)]) 
            The potential function defined on the grid
        units : class
            Class whose attributes are the fundamental constants hbar, e, m, c, etc.
        num_states : int
            number of states to be returns
        
    Output
        psi : np.array([psi_0(x), ..., psi_N(x)]) where N = NumStates
            Eigenfunctions of Hamiltonian
        E : np.array([E_0, ..., E_N]) 
            Eigenvalues of Hamiltonian
        
    Optional:
        num_states : int, default 15
            Dictates the number of states to solve for. Must be less than
            the number of spatial points - 2.
    """
        
    # Determine number of points in spacial grid
    N = len(x)
    dx = x[1]-x[0]
    
    #Reset num_states if the resolution of the space is less than the called for
    #  numer of states
    if num_states > N-2:
        print("Resolution too poor for requested number of states."+str(N-1)+
                "states returned.")
        num_states = N-1
    
    # Construct the Hamiltonian in position space
    H = solver_utils.make_hamiltonian(dx, V, units, boundary='hard_wall')
    
    # Compute eigenvalues and eigenfunctions:
    E, psi = linalg.eigh(H)
    
    # Truncate to the desired number of states
    E = E[:num_states]
    psi = psi[:,:num_states]
    
    # Hard walls are assumed at the boundary, so the wavefunction at the boundary
    #   must be zero. We enforce this boundary condition:
    psi = sp.insert(psi, 0, sp.zeros(num_states), axis = 0)
    psi = sp.insert(psi, len(x)-1, sp.zeros(num_states), axis = 0)
    
    # Normalize to unity:
    for i in range(num_states):
        psi[:,i] = psi[:,i] / sp.sqrt(sp.trapz( psi[:,i]*psi[:,i], x))
    
    # Take the transpose so that psi[i] is the ith eigenfunction:
    psi = psi.transpose()
    
    return psi, E

def apply_H_fft(f, x, V, units, q=0):
	"""Applies Hamiltonain to state using FFT methods."""
	N = len(x)
	dx = x[1]-x[0]
	k = 2*np.pi*np.fft.fftfreq(N, d=dx)
	K = np.fft.ifft((k+q)**2/2/units.m*np.fft.fft(f))
	return K + V*f

def solver_fft(x, V, units, Nstates=21, q=0):
    """Returns eigenstates and eigenenergies of one dimensional potentials
    define on grid using fourier methods to approximate the kinetic energy. 
    
    Assumes infinite walls at both ends of the problem space.
    
    Input
        x : np.array([x_i]) 
            The spacial grid points including the endpoints
        V : np.array([V(x_i)]) 
            The potential function defined on the grid
        units : class
            Class whose attributes are the fundamental constants hbar, e, m, c, etc.
        Nstates : int
            number of states to be returns
        q : np.float
            quasimomentum
        
    Output
        psi : np.array([psi_0(x), ..., psi_N(x)]) where N = NumStates
            Eigenfunctions of Hamiltonian
        E : np.array([E_0, ..., E_N]) 
            Eigenvalues of Hamiltonian
    """
        
    # Determine parameters from spacial grid
    N = len(x)
    dx = x[1]-x[0]
    L = N * dx

    # Determine k space parameters
    k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dx))

    # Kinetic energy in k space
    Tk = units.hbar**2 * (k+q)**2 / 2 / units.m

    # Transform to position space
    M_kn = np.exp(-1j * k[:, None] * x[None, :]) / np.sqrt(N)
    Tx = M_kn.conj().dot(Tk[:, None] * M_kn)

    # Construct the Hamiltonian in position space
    H = Tx + np.diag(V)
    assert np.allclose(H.transpose().conj(), H), "not Hermitian"
    
    # Compute eigenvalues and eigenfunctions:
    E, psi = linalg.eigh(H)
    
    E = E[:Nstates]
    psi = psi.transpose()
    psi = psi[:Nstates]
    # Normalize to unity:
    for i in range(Nstates):
        psi[i] = psi[i] / sp.sqrt(sp.trapz( psi[i]*psi[i], x))
    
    return psi, E