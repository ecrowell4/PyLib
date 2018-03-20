<<<<<<< HEAD
import scipy as sp
from scipy import linalg
=======
import scipy.linalg
>>>>>>> 78d9464ef861f91a76047dd8523cd06fee719e02
import numpy as np
from nlopy.quantum_solvers import solver_utils

def solver_1D(x, V, units, num_states=15):
    """Uses finite difference to discretize and solve for the eigenstates and 
    energy eigenvalues one dimensional potentials.
    
    The domain fed to this routine defines the problem space allowing
    non-uniform point density to pay close attention to particular parts of the
    potential. Assumes infinite walls at both ends of the problem space.
    
    Input
        x : np.array([x_i]) 
            The spacial grid points including the endpoints
        V : np.array([V(x_i)]) 
            The potential function defined on the grid
        units : class
            Class whose attributes are the fundamental constants hbar, e, m, c, etc.
        
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
    if num_states >= N-2:
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