def alpha_mm2(E, L, w=0):
    """Returns the magnetic-magnetic polarizability at frequency w.
    
    Input
    -----
    E : np.array(numstates)
        eigenenergies
    L : np.array(numstates, numstates)
        angular momentum matrix
    w : float
        frequency of optical field (zero for offresonance)
        
    Output
    ------
    alpha : complex float
        magnetic-magnetic polarizability
    """
    import numpy as np
    
    #We use natural units
    e = 1.
    m = 1.
    c = 1.
    gamma = e / 2 / m / c
    
    alpha = gamma**2 * (np.delete(L[0],0).conj().dot((1/np.delete((np.conjugate(E-E[0]) + w), 0))*np.delete(L[0],0))
                        + np.delete(L[0],0).conj().dot((1/np.delete((E-E[0] - w), 0))*np.delete(L[0],0)))
    
    return alpha