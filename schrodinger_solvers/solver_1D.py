
import numpy as np
import PyLib as pl

def solve_1D(x, V, units, NumStates=15):
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
    
    #Reset num_states if the resolution of the space is less than the called for
    #  numer of states
    if num_states >= N-2:
        print("Resolution too poor for requested number of states."+str(N-1)+
                "states returned.")
        num_states = N-1
    
    # Construct the Hamiltonian in position space
    H = pl.schrodinger_solvers.solver_utility.hamiltonian(x, V, units, boundary='hard_wall')
    
    # Compute eigenvalues and eigenfunctions:
    E, psi = np.linalg.eigh(H)
    
    # Truncate to the desired number of states
    E = E[:NumStates]
    psi = psi[:,:NumStates]
    
    # Hard walls are assumed at the boundary, so the wavefunction at the boundary
    #   must be zero. We enforce this boundary condition:
    psi = np.insert(psi, 0, np.zeros(NumStates), axis = 0)
    psi = np.insert(psi, len(x)-1, np.zeros(NumStates), axis = 0)
    
    # Normalize to unity:
    for i in range(NumStates):
        psi[:,i] = psi[:,i] / np.sqrt(np.trapz( psi[:,i]*psi[:,i], x))
    
    # Take the transpose so that psi[i] is the ith eigenfunction:
    psi = psi.transpose()
    
    return psi, E