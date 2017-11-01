def XMatrix(Psi, x, NumStates):
    """ Returns the position operator in the eigenenergy basis."""
    import numpy as np
    trapz = np.trapz
    conjugate = np.conjugate
    xx = np.zeros((NumStates, NumStates)) + 0j
    for i in range(NumStates):
        for k in range(NumStates):
            xx[k,i] = trapz( trapz( Psi[i] * x * conjugate(Psi[k]), x), x)
            
    return xx
    
def XMatrix_nojit(Psi, x, NumStates):
    xx = np.zeros((NumStates, NumStates)) + 0j
    for i in range(NumStates):
        for k in range(NumStates):
            tmp = np.zeros(len(x)) + 0j
            for l in range(len(x)):
                tmp[l] = np.trapz((Psi[i][:,l] * x * np.conjugate(Psi[k][:,l])), x)
            xx[k,i] = np.trapz(tmp, x)
            
    return xx
    
def sqrarray1(x):
    n = len(x)
    for i in range(n):
        x[i]=x[i]**2
    return x
    
def sqr(x):
    return x**2
    
    