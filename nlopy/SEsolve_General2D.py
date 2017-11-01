import numpy as np
import PyLib as pl
from concurrent import futures

def SEsolve_General2D_Parallel(x, y, Vm, fmxy, fmx, fmy, NumStates = 15, nounits=True):
    """Uses finite difference to discretize and solve for the eigenstates and 
    energy eigenvalues two dimensional problems.  We will assume the problem
    space to be an N+2 by N+2 square grid. Note that this implies that Xnm = Ynm.
    All derivatives are central differences.
    
    Assumes infinite walls at both ends of the problem space.
    
    All functions passed in (potential and the three coefficient functions) are
    assumed to be defined on a 2D grid with y values along the row (i.e. a 
    numpy meshgrid with 'ij' indexing).
    
    Input:
        x,y = np.array([x_i, y_i]) the spacial grid points including the endpoints
               not meshed
        V = np.array([V(x_i, y_i)]) the potential function defined on the grid
        am = np.array([a[x_i,y_i]]) the coefficient function for pxpy term
        fmx = np.array([bx[x_i, y_i]]) the coefficient function for linear px term
        fmy = np.array([by[x_i, y_i]]) coefficient funciton for linear py term
        
    Units:
        [x] = cm
        [V] = eV
        
    Output:
        psi = np.array([psi_0(x), ..., psi_N(x)]) where N = NumStates
        E = np.array([E_0, ..., E_N]) the energy eigenvalues in eV
        xx = np.array([<i|x|j>]) the transition moments
        
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
    N = len(x)-2
    
    #Reset NumStates if the resolution of the space is less than the called for
    #  numer of states
    if NumStates >= N:
        print("Resolution too poor for requested number of states."+str(N-1)+
                "states returned.")
        NumStates = N-1
    
    #The vector dx represents the grid spacing. For our purposes we use a uniform
    # grid, so we actually only need dx[0]
    x = np.array(x)
    dx = x[1:]-x[:-1]  
    
#==============================================================================
# In this section we form the blocks that will go along the diagonal of our 
# Hamiltonian.  Concurrent futures is used to form the blocks in parallel.
    
    pool = futures.ThreadPoolExecutor(20)
    
    def Bmat( i ):
        """
        Return the ith diagonal block
        """
        return np.diag(2 * np.ones(N) / dx[0]**2)\
        + np.diag( -np.ones(N-1) / (2*dx[0]**2), k=1)\
        + np.diag( -np.ones(N-1) / (2*dx[0]**2), k=-1)\
        + np.diag(-1j*fmx[i+1, 1:-2] / (2*dx[0]), k=1)\
        + np.diag(1j*fmx[i+1, 2:-1] / (2*dx[0]), k=-1)\
        + np.diag(Vm[i+1, 1:-1])
    
    def Amat( i ):
        """
        Return the ith upper diagonal block
        """
        return np.diag(-np.ones(N) / (2*dx[0]**2))\
        + np.diag( -1j*fmy[i+1][1:-1] / (2*dx[0]))\
        + np.diag( -fmxy[:,i+1][1:-2] / (4*dx[0]**2), k=1)\
        + np.diag( fmxy[:,i+1][2:-1] / (4*dx[0]**2), k=-1)
        
    def Cmat( i ):
        """
        Return the ith lower diagonal block
        """
        return np.diag(-np.ones(N) / (2*dx[0]**2))\
        + np.diag( 1j*fmy[i+2][1:-1] / (2*dx[0]) )\
        + np.diag(fmxy[:,i+2][1:-2] / (4*dx[0]**2), k=1 )\
        + np.diag( -fmxy[:,i+2][2:-1] / (4*dx[0]**2), k=-1)

    B = np.asarray(list( pool.map( Bmat, np.arange( N ) ) ) )
    A = np.asarray(list( pool.map( Amat, np.arange( N-1 ) ) ) )
    C = np.asarray(list( pool.map( Cmat, np.arange( N-1 ) ) ) )

    # Form the hamiltonian out of the blocks
    H = pl.BlockDiag.diagblock(B) + pl.BlockDiag.diagblock(A, k=1) + pl.BlockDiag.diagblock(C, k=-1)
    
    # We ensure H is Hermitian
    H = 0.5 * (H + np.transpose(np.conjugate(H)))
    
    # Diagonalize
    E, psi = np.linalg.eigh(H)
    
    # Keep only the desired number of states    
    E = E[:NumStates]
    psi = psi[:,:NumStates]
    
    # Take transpose so that psi[i] is the ith state wavefunction
    psi = psi.transpose()
    
    #Go back to 2D grid representation of the states
    Psi = np.zeros((NumStates, N, N)) + 0j
    for i in range(NumStates):
        for l in range(N):
            Psi[i][l] = psi[i][l*N:(l+1)*N]
            
    #Insert boundary values (i.e. zeros at x,y = 0, L)    
    Psi = np.insert(Psi, 0, np.zeros(N), axis = 1)
    Psi = np.insert(Psi, 0, np.zeros(N+1), axis = 2)            
    Psi = np.insert(Psi, N+1, np.zeros(N+1), axis = 1)
    Psi = np.insert(Psi, N+1, np.zeros(N+2), axis = 2)
    
    # Normalize to unity   
    Psi = np.transpose(np.transpose(Psi) 
    / np.sqrt(np.trapz( np.trapz( Psi*np.conjugate(Psi), x), x )))
    
    # Compute transition moments for a square problem space => xx = yy, so we only
    # compute one of them here
    xx = pl.XMatrix.XMatrix(Psi, x, NumStates)

    return Psi, E, xx