import scipy as sp
import numpy as np
from mpmath import mp
from nlopy.quantum_solvers import solver_utils

mp.dps = 15
def solver_mp(x, V, units, num_states=15):
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
    E, psi = mp.eigh(mp.matrix(H))
    
    # Truncate to the desired number of states
    E = E[:num_states]
    psi = psi[:,:num_states]
    
    psi = sp.insert(np.array(psi.tolist()), 0, sp.zeros(num_states), axis = 0)
    psi = sp.insert(np.array(psi.tolist()), len(x)-1, sp.zeros(num_states), axis = 0)
    for i in range(num_states):
        psi[:,i] = psi[:,i] / sp.sqrt(sp.trapz( psi[:,i]*psi[:,i], x))
    
    psi = np.array(psi.tolist())
    psi = psi.transpose()
    
    E = np.array(E.tolist())
    
    return psi, E