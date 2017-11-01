def RandomEnergy(N):
    """Returns an ordered array of length N.  The first two elements are
    0 and 1, respectively.
    
    Parameters
    ----------
    INPUT
    N : int
        number of states after truncation (including ground state)
        
    OUTPUT
    E : numpy array object, shape = (N,)
        ordered array of normalized energies
    """
    
    import numpy as np
    
    E = np.random.random(N-1)
    E = E[E.argsort()]
    E = np.insert(E, 0, 0)
    E /= E[1]
    
    return E