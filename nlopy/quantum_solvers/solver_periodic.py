import numpy as np
from nlopy.quantum_solvers import solver_utils

def solver_periodic(x, V, units, num_states=15, Bfield=None):
    """Uses finite difference to discretize and solve for the eigenstates and 
    energy eigenvalues of one dimensional periodic potentials.
        
    Input:
        x : np.array([x_i]) 
            the spacial grid points including the endpoints
        V : np.array([V(x_i)]) 
            the potential function defined on the grid
        units : Bool
            fundamental constants
        Bfield : np.float
            magnetic field strength
        
    Units:
        [x] = cm
        [V] = eV
        
    Output:
        psi = np.array([psi_0(x), ..., psi_N(x)]) where N = NumStates
        E = np.array([E_0, ..., E_N]) the energy eigenvalues in eV
        L = np.array([<i|L|j>]) the angular momentum matrix
        
    Optional:
        NumStates = 15; Dictates the number of states to solve for.
    """
    
    #The first entry in the x array is the regularization length, which should
    #not be explicitly included in the potential to arrive at the appropriate 
    #boundary condition. The last entry is used in the definition of the 
    #derivative but does not hold a function value.
    N = len(x)
    dx = x[1]-x[0]
    
    #Reset NumStates if the resolution of the space is less than the called for
    #  numer of states
    if num_states >= N-2:
        print("Resolution too poor for requested number of states."+str(N-1)+
                "states returned.")
        num_states = N-1
    
    # Construct the Hamiltonian
    H = solver_utils.make_hamiltonian(dx, V, units, boundary='periodic', Bfield=Bfield)
    
    #Enforce periodic boundary conditions (this is one of two conditions):
    #H[0,N-2] = -1 / (2*dx**2) 
    #H[N-2,0] = -1 / (2*dx**2) 

    #Diagonalize
    E, psi = np.linalg.eigh(H)

    E = E[:num_states]
    psi = psi[:,:num_states]
    
    psi = psi.transpose()
    
    #Append the last term in wave function (second periodic condition)   
    Psi = np.zeros((num_states, N)) + 0j
    for i in range(num_states):
        Psi[i] = np.insert(psi[i], N-1, psi[i][0])
    
    #Normalize to unity
    for i in range(num_states):
        Psi[i] = Psi[i] / np.sqrt(np.trapz( Psi[i]*np.conjugate(Psi[i]), x))

    return Psi, E