import numpy as np

def PeriodicSEsolve(x, V, NumStates = 15, nounits=True):
    """Uses finite difference to discretize and solve for the eigenstates and 
    energy eigenvalues of one dimensional periodic potentials.
        
    Input:
        x = np.array([x_i]) the spacial grid points including the endpoints
        V = np.array([V(x_i)]) the potential function defined on the grid
        
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
    
    hbar = 6.58211928E-16 #eV.s
    m = 0.51099891E6 / (2.99792458E10)**2 #eV.s^2/cm^2
    if nounits == True:
        hbar = 1.
        m = 1.
        
    #The first entry in the x array is the regularization length, which should
    #not be explicitly included in the potential to arrive at the appropriate 
    #boundary condition. The last entry is used in the definition of the 
    #derivative but does not hold a function value.
    N = len(x)
    
    #Reset NumStates if the resolution of the space is less than the called for
    #  numer of states
    if NumStates >= N:
        print("Resolution too poor for requested number of states."+str(N-1)+
                "states returned.")
        NumStates = N-1
    
    #Form the matrix representation of the potential in the x basis
    V = np.diag(V[1:])
    
    x = np.array(x)
    dx = x[1]-x[0]  
    
    #The kinetic energy matrix 
    T = ((-2 / (dx**2))*np.eye(N-1) +
        (1 / (dx**2))*np.eye(N-1,k=1) +
        (1 / (dx**2))*np.eye(N-1,k=-1))
        
    H = -hbar**2 / ( 2 * m ) * T + V
    
    
    #Enforce periodic boundary conditions (this is one of two conditions):
    H[0,N-2] = -1 / (2*dx**2) 
    H[N-2,0] = -1 / (2*dx**2) 
    
    #Diagonalize
    E, psi = np.linalg.eigh(H)
    E = E[:NumStates]
    psi = psi[:,:NumStates]
    
    psi = psi.transpose()
    
    #Append the last term in wave function (second periodic condition)   
    Psi = np.zeros((NumStates, N)) + 0j
    for i in range(NumStates):
        Psi[i] = np.insert(psi[i], N-1, psi[i][0])
    
    #Normalize to unity
    for i in range(NumStates):
        Psi[i] = Psi[i] / np.sqrt(np.trapz( Psi[i]*np.conjugate(Psi[i]), x))
    

    L = np.zeros((NumStates,NumStates))+0j
    for n in range(NumStates):
        for m in range(NumStates):
            L[n,m] = -1j * hbar * np.trapz( np.conjugate(Psi[n]) * (np.gradient(Psi[m])/dx), x)

    return Psi, E, L