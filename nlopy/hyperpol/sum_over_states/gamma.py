import numpy as np
from numba import jit
from nlopy.hyperpol.sum_over_states import sos_utils

def gamma_eeee_(E, X, units, omega=np.zeros(3), n=0, damping=False):
    """Compute the all electric second order hyperpolarizability for
    a system "in nth state".

    Input
        E : np.array
            eigenenergies of unperturbed system
        X : np.array
            transition matrix of unperturbed system
        omega : np.array(3)
            incident field frequencies
        n : int
            "state of system". Specifically, it is the unperturbed state
            that is closest to the actual eigenstate in presence of field.

    Output
        gamma : complex
            All electric second hyperpolarizability
    """
    num_states = len(E)
    X = X - X[n,n]*np.eye(num_states)
    E = E - E[n]
    L = None
    I = None
    if damping == True:
        Gamma = (units.c**3 /10)*(2 / 3 / units.hbar) * (E / units.c)**3 * units.e**2 * abs(xx[:,0])**2
        E = E - 1j * Gamma / units.hbar
    gamma_eeee = (
        sos_utils.permute_gamma_4op_terms(
            sos_utils.gamma_term11, X, X, X, X, E, omega, units, gamma_type='eeee', n=n)
        + sos_utils.permute_gamma_4op_terms(
            sos_utils.gamma_term12, X, X, X, X, E, omega, units, gamma_type='eeee', n=n)
        + sos_utils.permute_gamma_4op_terms(
            sos_utils.gamma_term13, X, X, X, X, E, omega, units, gamma_type='eeee', n=n)
        + sos_utils.permute_gamma_4op_terms(
            sos_utils.gamma_term14, X, X, X, X, E, omega, units, gamma_type='eeee', n=n)
        - sos_utils.permute_gamma_4op_terms(
            sos_utils.gamma_term21, X, X, X, X, E, omega, units, gamma_type='eeee', n=n)
        - sos_utils.permute_gamma_4op_terms(
            sos_utils.gamma_term22, X, X, X, X, E, omega, units, gamma_type='eeee', n=n)
        - sos_utils.permute_gamma_4op_terms(
            sos_utils.gamma_term23, X, X, X, X, E, omega, units, gamma_type='eeee', n=n)
        - sos_utils.permute_gamma_4op_terms(
            sos_utils.gamma_term24, X, X, X, X, E, omega, units, gamma_type='eeee', n=n)
        )
    return gamma_eeee 

def gamma_eeee(E, X, units, omega=np.zeros(3), n=0, damping=False):
    """Compute the all electric second order hyperpolarizability for
    a system "in nth state".

    Input
        E : np.array
            eigenenergies of unperturbed system
        X : np.array
            transition matrix of unperturbed system
        omega : np.array(3)
            incident field frequencies
        n : int
            "state of system". Specifically, it is the unperturbed state
            that is closest to the actual eigenstate in presence of field.

    Output
        gamma : complex
            All electric second hyperpolarizability
    """
    num_states = len(E)
    X = X - X[n,n]*np.eye(num_states)
    E = E - E[n]
    L = None
    I = None
    if damping == True:
        Gamma = (units.c**3 /10)*(2 / 3 / units.hbar) * (E / units.c)**3 * units.e**2 * abs(xx[:,0])**2
        E = E - 1j * Gamma / units.hbar
    gamma_eeee = (
        sos_utils.permute_gamma_terms_123(
            sos_utils.gamma_term11, E, X, L, I, omega, units, gamma_type='eeee', n=n)
        + sos_utils.permute_gamma_terms_123(
            sos_utils.gamma_term12, E, X, L, I, omega, units, gamma_type='eeee', n=n)
        + sos_utils.permute_gamma_terms_123(
            sos_utils.gamma_term13, E, X, L, I, omega, units, gamma_type='eeee', n=n)
        + sos_utils.permute_gamma_terms_123(
            sos_utils.gamma_term14, E, X, L, I, omega, units, gamma_type='eeee', n=n)
        - sos_utils.permute_gamma_terms_123(
            sos_utils.gamma_term21, E, X, L, I, omega, units, gamma_type='eeee', n=n)
        - sos_utils.permute_gamma_terms_123(
            sos_utils.gamma_term22, E, X, L, I, omega, units, gamma_type='eeee', n=n)
        - sos_utils.permute_gamma_terms_123(
            sos_utils.gamma_term23, E, X, L, I, omega, units, gamma_type='eeee', n=n)
        - sos_utils.permute_gamma_terms_123(
            sos_utils.gamma_term24, E, X, L, I, omega, units, gamma_type='eeee', n=n)
        )
    return gamma_eeee 

def gamma_mmmm(E, L, I, units, omega=np.zeros(3), n=0, damping=False):
    """Compute the all magnetic second order hyperpolarizability for
    a system "in nth state".

    Input
        E : np.array
            eigenenergies of unperturbed system
        X : np.array
            transition matrix of unperturbed system
        omega : np.array(3)
            incident field frequencies
        n : int
            "state of system". Specifically, it is the unperturbed state
            that is closest to the actual eigenstate in presence of field.

    Output
        gamma : complex
            All electric second hyperpolarizability
    """
    num_states = len(E)
    L = L - X[n,n]*np.eye(num_states)
    E = E - E[n]
    X = None
    if damping == True:
        Gamma = (units.c**3 /10)*(2 / 3 / units.hbar) * (E / units.c)**3 * units.e**2 * abs(xx[:,0])**2
        E = E - 1j * Gamma / units.hbar
    gamma_eeee = (
        sos_utils.permute_gamma_terms_123(
            sos_utils.gamma_term11, E, X, L, I, omega, units, gamma_type='eeee', n=n)
        + sos_utils.permute_gamma_terms_123(
            sos_utils.gamma_term12, E, X, L, I, omega, units, gamma_type='eeee', n=n)
        + sos_utils.permute_gamma_terms_123(
            sos_utils.gamma_term13, E, X, L, I, omega, units, gamma_type='eeee', n=n)
        + sos_utils.permute_gamma_terms_123(
            sos_utils.gamma_term14, E, X, L, I, omega, units, gamma_type='eeee', n=n)
        - sos_utils.permute_gamma_terms_123(
            sos_utils.gamma_term21, E, X, L, I, omega, units, gamma_type='eeee', n=n)
        - sos_utils.permute_gamma_terms_123(
            sos_utils.gamma_term22, E, X, L, I, omega, units, gamma_type='eeee', n=n)
        - sos_utils.permute_gamma_terms_123(
            sos_utils.gamma_term23, E, X, L, I, omega, units, gamma_type='eeee', n=n)
        - sos_utils.permute_gamma_terms_123(
            sos_utils.gamma_term24, E, X, L, I, omega, units, gamma_type='eeee', n=n)
        )
    return gamma_eeee 

def gamma_mmmm_(E, L, I, omega, units, n=0, includeA2=True, includeCovar=True, damping=False):
    """Compute diagonal component of gamma_mmmm with or without A^2 term. 
    
    Input
        E : np.array
            eigenenergies of unperturbed system
        xx : np.array
            transition matrix of unperturbed system
        units : class
            fundamental constants
        sq_term : bool
            If true, then return gamma with A^2 term included in perturbation.
    
    Output
        gamma_eeee : complex
            second hyperpolarizability
    """
    
    Del = np.delete

    # assert consisten dimensions
    assert len(E) == len(L[0])
    
    # determine number of eigenstates fo be used in computing beta
    num_states = len(E)
    
    # Take all mu -> bar{mu}
    L = L - L[n,n]*np.eye(num_states)
    I = I - I[n,n]*np.eye(num_states)
    
    # Take all Em -> Enm
    E = E - E[0]
    
    # Include damping if damping is true
    if damping == True:
        # Define damping coeffs
        Gamma = (units.c**3 /10)*(2 / 3 / units.hbar) * (E / units.c)**3 * units.e**2 * abs(xx[:,0])**2
        
        # Incorporate into energies
        E = E - 1j * Gamma / units.hbar
    
    # compute gamma term by term
    gamma = (sos_utils.permute_gamma_terms_123(sos_utils.gamma_term11_m, E, X, L, I, omega, units) 
    + sos_utils.permute_gamma_terms_123(sos_utils.gamma_term12_m, E, X, L, I, omega, units)
    + sos_utils.permute_gamma_terms_123(sos_utils.gamma_term13_m, E, X, L, I, omega, units)
    + sos_utils.permute_gamma_terms_123(sos_utils.gamma_term14_m, E, X, L, I, omega, units)
    + sos_utils.permute_gamma_terms_123(sos_utils.gamma_term21_m, E, X, L, I, omega, units)
    + sos_utils.permute_gamma_terms_123(sos_utils.gamma_term22_m, E, X, L, I, omega, units)
    + sos_utils.permute_gamma_terms_123(sos_utils.gamma_term23_m, E, X, L, I, omega, units)
    + sos_utils.permute_gamma_terms_123(sos_utils.gamma_term24_m, E, X, L, I, omega, units))
    
    if includeA2 == True:
        gamma +=  (sos_utils.permute_gamma_terms_123(sos_utils.gamma_term31_m, E, X, L, I, omega, units) 
        + sos_utils.permute_gamma_terms_123(sos_utils.gamma_term32_m, E, X, L, I, omega, units)
        + sos_utils.permute_gamma_terms_123(sos_utils.gamma_term33_m, E, X, L, I, omega, units)
        + sos_utils.permute_gamma_terms_123(sos_utils.gamma_term34_m, E, X, L, I, omega, units)
        + sos_utils.permute_gamma_terms_123(sos_utils.gamma_term35_m, E, X, L, I, omega, units)
        + sos_utils.permute_gamma_terms_123(sos_utils.gamma_term36_m, E, X, L, I, omega, units))
    
    if includeCovar == True:
        gamma += (sos_utils.permute_gamma_terms_123(sos_utils.gamma_term41_m, E, X, L, I, omega, units)
        + sos_utils.permute_gamma_terms_123(sos_utils.gamma_term42_m, E, X, L, I, omega, units)
        + sos_utils.permute_gamma_terms_123(sos_utils.gamma_term43_m, E, X, L, I, omega, units))
        
    if includeA2==True and includeCovar==True:
        gamma += (sos_utils.permute_gamma_terms_123(sos_utils.gamma_term51_m, E, X, L, I, omega, units)
        + sos_utils.permute_gamma_terms_123(sos_utils.gamma_term52_m, E, X, L, I, omega, units))
    
    return gamma


@jit(nopython=True)
def gamma_eeee_densityMatrix(rho0:float, X:complex, E:float, omega:float, gamma_damp:float)->complex:
    """Returns the second hyperpolarizability as computed from the
    density matrix formalism. See Boyd's text for the equations.

    Input
        rho0 : np.array
            zeroth order density matrix
        X : np.array
            transition matrix
        E : np.array
            eigenenergies
        omega : np.array
            incident field frequencies
        gamma_damp : np.array
            damping constants gamma_nl
        units : class
            fundamental constants

    Output
        gamma_eeee : complex
            diagonal element of gamma
    """
    N:float = len(rho0)
    gamma:complex = 0+0j
    for l in range(N):
        for n in range(N):
            for m in range(N):
                for v in range(N):
                    if n!=m and m!=v and m!=l:
                        if rho0[m,m]-rho0[l,l]!= 0:
                            print(l,n,m,v)
                        gamma += ((rho0[m,m] - rho0[l,l])*(
                            X[m,n]*X[n,v]*X[v,l]*X[l,m]/(E[n]-E[m]-1j*gamma_damp[n,m])/(E[v]-E[m]-1j*gamma_damp[v,m])/(E[l]-E[m]-1j*gamma_damp[l,m])))
                    if n!=m and m!=v and l!=v:
                        if rho0[l,l]-rho0[v,v]!=0:
                            print(l,n,m,v)
                        gamma -= ((rho0[l,l] - rho0[v,v])*(
                            X[m,n]*X[n,v]*X[v,l]*X[l,m]/(E[n]-E[m]-1j*gamma_damp[n,m])/(E[v]-E[m]-1j*gamma_damp[v,m])/(E[v]-E[l]-1j*gamma_damp[v,l])))
                    if n!=m and n!=v and l!=v:
                        if rho0[v,v]-rho0[l,l]!= 0:
                            print(l,n,m,v)
                        gamma -= (rho0[v,v] - rho0[l,l])*(
                            X[m,n]*X[n,v]*X[v,l]*X[l,m]/(E[n]-E[m]-1j*gamma_damp[n,m])/(E[n]-E[v]-1j*gamma_damp[n,v])/(E[l]-E[v]-1j*gamma_damp[l,v]))
                    if n!=m and v!=n and l != n:
                        if rho0[l,l]-rho0[n,n]!=0:
                            print('third', l,n,m,v)
                        gamma += ((rho0[l,l] - rho0[n,n])*(
                            X[m,n]*X[n,v]*X[v,l]*X[l,m]/(E[n]-E[m]-1j*gamma_damp[n,m])/(E[n]-E[v]-1j*gamma_damp[n,v])/(E[n]-E[l]-1j*gamma_damp[n,l])))
                        
    return gamma
    

