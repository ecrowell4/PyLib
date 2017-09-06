
import numpy as np

def SEsolve(x, V, NumStates = 15, nounits = True):
    """Uses finite difference to discretize and solve for the eigenstates and 
    energy eigenvalues one dimensional potentials.
    
    The domain fed to this routine defines the problem space allowing
    non-uniform point density to pay close attention to particular parts of the
    potential. Assumes infinite walls at both ends of the problem space.
    
    Input:
        x = np.array([x_i]) the spacial grid points including the endpoints
        V = np.array([V(x_i)]) the potential function defined on the grid
        
    Units:
        [x] = cm
        [V] = eV
        
    Output:
        psi = np.array([psi_0(x), ..., psi_N(x)]) where N = NumStates
        E = np.array([E_0, ..., E_N]) the energy eigenvalues in eV
        xx = np.array([<i|x|j>]) the transition moments
        
    Optional:
        NumStates = 15; Dictates the number of states to solve for.
        nounits = False; 
    """
    
    hbar = 6.58211928E-16 #eV.s
    m = 0.51099891E6 / (2.99792458E10)**2 #eV.s^2/cm^2
    if nounits == True:
        hbar = 1.
        m = 1.
        
    #The first entry in the x array is the regularization length, which should
    #not be explicitly included in the potential to arrive at the appropriate 
    #boundary condition. The last entry is used in the definition of the 
    #derivative but does not hold a function value.
    N = len(x)-2
    
    #Reset NumStates if the resolution of the space is less than the called for
    #  numer of states
    if NumStates >= N:
        print("Resolution too poor for requested number of states."+str(N-1)+
                "states returned.")
        NumStates = N-1
    
    #Form the matrix representation of the potential in the x basis
    V = np.diag(V[1:-1])
    
    #The vector h represents the grid spacing
    x = np.array(x)
    h = x[1:]-x[:-1]   
    
    #The kinetic energy matrix tailored by an arbitrary, ordered, domain array
    T = ( np.diag(-2 / (h[1:]*h[:-1])) +
        np.diag(1 / (h[1:-1]*(h[1:-1]/2.+h[:-2]/2)),k=1) +
        np.diag(1 / (h[1:-1]*(h[2:]/2.+h[1:-1]/2.)),k=-1))

    H = -hbar**2 / ( 2 * m ) * T + V
    
    #Determine the transformation matrix which will make the Hamiltonian matrix
    #   symmetric
    D = np.zeros(N)
    D[0] = 1.
    for i in range(1,N):
        D[i] = (D[i-1]**2 * H[i,i-1]/H[i-1,i])**0.5
    
    Hdiag = np.dot(np.dot(np.diag(1./D), H), np.diag(D))
    
    E, psi = np.linalg.eigh(Hdiag)
    
    E = E[:NumStates]
    psi = psi[:,:NumStates]
    
    #Transform each eigenstate back into x space and normalize
    for i in range(NumStates):
        psi[:,i] = np.dot(np.diag(D), psi[:,i])
    psi = np.insert(psi, 0, np.zeros(NumStates), axis = 0)
    psi = np.insert(psi, len(x)-1, np.zeros(NumStates), axis = 0)
    for i in range(NumStates):
        psi[:,i] = psi[:,i] / np.sqrt(np.trapz( psi[:,i]*psi[:,i], x))
    
    xx = [np.trapz(x * psi[:,l] * psi[:,p], x )
            for l in range(NumStates) for p in range(NumStates)]
    xx = np.reshape(xx,(NumStates,NumStates))
    
    psi = psi.transpose()
    
    return psi, E, xx