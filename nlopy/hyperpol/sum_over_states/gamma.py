import numpy as np
from nlopy.hyperpol.sum_over_states import sos_utils

def gamma_mmmm(L, I, E, omega, units, n=0, includeA2=True, includeCovar=True, damping=False):
    """Compute diagonal component of gamma_eeee with or without A^2 term. 
    
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
    L = L - L[0,0]*np.eye(15)
    I = I - I[0,0]*np.eye(15)
    
    # Take all Em -> Enm
    E = E - E[0]
    
    # Include damping if damping is true
    if damping == True:
        # Define damping coeffs
        Gamma = (units.c**3 /10)*(2 / 3 / units.hbar) * (E / units.c)**3 * units.e**2 * abs(xx[:,0])**2
        
        # Incorporate into energies
        E = E - 1j * Gamma / units.hbar
    
    # compute gamma term by term
    gamma = (sos_utils.permute_gamma_terms(sos_utils.gamma_term11, L, I, E, omega, units) 
    + sos_utils.permute_gamma_terms(sos_utils.gamma_term12, L, I, E, omega, units)
    + sos_utils.permute_gamma_terms(sos_utils.gamma_term13, L, I, E, omega, units)
    + sos_utils.permute_gamma_terms(sos_utils.gamma_term14, L, I, E, omega, units)
    + sos_utils.permute_gamma_terms(sos_utils.gamma_term21, L, I, E, omega, units)
    + sos_utils.permute_gamma_terms(sos_utils.gamma_term22, L, I, E, omega, units)
    + sos_utils.permute_gamma_terms(sos_utils.gamma_term23, L, I, E, omega, units)
    + sos_utils.permute_gamma_terms(sos_utils.gamma_term24, L, I, E, omega, units))
    
    if includeA2 == True:
        gamma +=  (sos_utils.permute_gamma_terms(sos_utils.gamma_term31, L, I, E, omega, units) 
        + sos_utils.permute_gamma_terms(sos_utils.gamma_term32, L, I, E, omega, units)
        + sos_utils.permute_gamma_terms(sos_utils.gamma_term33, L, I, E, omega, units)
        + sos_utils.permute_gamma_terms(sos_utils.gamma_term34, L, I, E, omega, units)
        + sos_utils.permute_gamma_terms(sos_utils.gamma_term35, L, I, E, omega, units)
        + sos_utils.permute_gamma_terms(sos_utils.gamma_term36, L, I, E, omega, units))
    
    if includeCovar == True:
        gamma += (sos_utils.permute_gamma_terms(sos_utils.gamma_term41, L, I, E, omega, units)
        + sos_utils.permute_gamma_terms(sos_utils.gamma_term42, L, I, E, omega, units)
        + sos_utils.permute_gamma_terms(sos_utils.gamma_term43, L, I, E, omega, units))
        
    if includeA2==True and includeCovar==True:
        gamma += (sos_utils.permute_gamma_terms(sos_utils.gamma_term51, L, I, E, omega, units)
        + sos_utils.permute_gamma_terms(sos_utils.gamma_term52, L, I, E, omega, units))
    
    return gamma

