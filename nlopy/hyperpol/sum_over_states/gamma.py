import numpy as np
from nlopy.hyperpol.sum_over_states import sos_utils
#from numba import jit

    
def gamma_mmmm(E, L, I, units, canonical=False):
    """Returns the canonical, diagonal tensor component of the quadratic mangetic polarizability 
    beta at imput frequencies omega1 and omega2:
        m = beta B(omega1) B(omega2)
        
    INPUT
        E : np.array(numstates)
            eigenenergies of unperturbed system
        L : np.array((numstates,numstates))
            angular momentum matrix of unperturbed system
        I : np.array((numstates, numstates))
            moment of inertia matrix of unperturbed system
        units : class
        	class object containing fundamental constants
        canonical : bool
            if True, use canonical angular momentum in SOS expression
    
    OUTPUT
        Gamma : complex
            first magnetic hyperpolarizability
    """
    
    N = len(E)
    Gamma = 0
    i = 1
    j = 1
    l = 1
    Lbar = L - L[0,0]
    while i < N:
        while j < N:
            while l < N:
                Gamma += L[0,i]*Lbar[i,j]*Lbar[j,l]*L[l,0] / ((E[i]-E[0])*(E[j]-E[0])*(E[l]-E[0]))
                l += 1   
            j += 1
        i += 1
    
    while i < N:
        while j < N:
            Gamma -= L[0,i]*L[i,0]*L[0,j]*L[j,0] / ((E[j]-E[0])**2 * (E[i]-E[0]))
    
    Gamma *= 4*units.g**4 / units.hbar**3
    
    if canonical is True:
        return Gamma
    else:
        i = 1
        j = 1
        while i < N:
            while j < N:
                if i != j:
                    Gamma += (2*I*units.g**4 / units.hbar**2) * L[0,i]*L[j,0] / ((E[i]-E[0])*(E[j]-E[0]))
                j += 1
            i += 1
        return Gamma

def gamma_term(E, L, indices):
    """Returns the term of gamma corresponding to
    n, m, l = indices[0], indices[1], indices[2].

    Input
        E : np.array
            eigenenergies
        L : np.array
            angular momentum matrix
        indices : array [i,j,l]
            array of indices

    Output
        gamma_nml : np.array
            a single term of the SOS expression for gamma
    """
    i,j,l = indices
    Lbar = L - L[0,0]

    return 4*units.g**4 / units.hbar**3 * (
        L[0,i]*Lbar[i,j]*Lbar[j,l]*L[l,0] / ((E[i]-E[0])*(E[j]-E[0])*(E[l]-E[0]))
        -L[0,i]*L[i,0]*L[0,j]*L[j,0] / ((E[j]-E[0])**2 * (E[i]-E[0]))
        )
