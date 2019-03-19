import numpy as np
from nlopy.hyperpol.sum_over_states import sos_utils

def gamma_eeee(E, xx, ein=[0,0,0]):
    """Calculates the first hyperpolarizability, beta, as a function of two 
    input photon energies and complex molecular transition and energy 
    information. Calculates the one tensor component specified.
    Input: 
        xx = [ mu_x, mu_y, mu_z ] the dipole moment matrices in cartesian
            coordinates normalized to the maximum transition strength x_10(max)
            where mu_x = [<i|e x|j>]
        E = [E_nm - i/2 Gamma_nm] the electronic eigen frequencies 
            corresponding to the transitions from the ith eigen state to the
            ground state 
            OR
            [E_n0] as a vector
        *The sizes of xx and E represent the number of states considered in
            this calculation and therefore must be consistant.*
        ijk = [0,1,2] the tensor components of beta where x=0, y=1, z=2
    Option:
        Calculate dispersion:
        ein = hbar*[w_j, w_k] input photon energies, which implies output 
            energy of (ein[0] + ein[1])
        Start and end in a given quantum state
        start = int some integer state which is represented by the xi and E info
    Output:
        beta = beta_ijk(-w_sigma; w[0], w[1])
    """

    e = 1
    hbar = 1

    #Find the number of electronic states supplied
    NumStates = len(xx[0])
    
    #Check for consistancy between transitions and energies
    if NumStates != len(E):
        print('Err: Inconsistant electronic state information.')
        return 0
    
    
    #Take all mu -> bar{mu}
    xx = xx - xx[0,0] * np.eye(NumStates)
    
    #Take all E -> E - E[0]
    E = E - E[0]
    
    # get rid of elements that should be zero
    for i in range(NumStates):
        if np.allclose(xx[i,i], 0)==True:
            xx[i,i] = 0
       
    ein_sigma = sum(ein)
    
    #Calculate beta
    gamma = e**4 / 6 / hbar**3 * ( (xx[0,1:] / (E[1:] - ein_sigma)).dot(xx[1:,1:].dot((xx[1:,1:]/(E[1:] - ein[0] - ein[1])).dot(xx[1:,0]/(E[1:] - ein[0]))))
    + (xx[0,1:] / (E[1:] - ein_sigma)).dot(xx[1:,1:].dot((xx[1:,1:]/(E[1:] - ein[0] - ein[2])).dot(xx[1:,0]/(E[1:] - ein[0]))))
    + (xx[0,1:] / (E[1:] - ein_sigma)).dot(xx[1:,1:].dot((xx[1:,1:]/(E[1:] - ein[1] - ein[0])).dot(xx[1:,0]/(E[1:] - ein[1]))))
    + (xx[0,1:] / (E[1:] - ein_sigma)).dot(xx[1:,1:].dot((xx[1:,1:]/(E[1:] - ein[1] - ein[2])).dot(xx[1:,0]/(E[1:] - ein[1]))))
    + (xx[0,1:] / (E[1:] - ein_sigma)).dot(xx[1:,1:].dot((xx[1:,1:]/(E[1:] - ein[2] - ein[0])).dot(xx[1:,0]/(E[1:] - ein[2]))))
    + (xx[0,1:] / (E[1:] - ein_sigma)).dot(xx[1:,1:].dot((xx[1:,1:]/(E[1:] - ein[2] - ein[1])).dot(xx[1:,0]/(E[1:] - ein[2]))))
    + (xx[0,1:] / (E[1:].conjugate() + ein[2])).dot(xx[1:,1:].dot((xx[1:,1:]/(E[1:].conjugate() + ein_sigma - ein[1])).dot(xx[1:,0]/(E[1:] - ein[1]))))
    + (xx[0,1:] / (E[1:].conjugate() + ein[0])).dot(xx[1:,1:].dot((xx[1:,1:]/(E[1:].conjugate() + ein_sigma - ein[2])).dot(xx[1:,0]/(E[1:] - ein[2]))))
    + (xx[0,1:] / (E[1:].conjugate() + ein[0])).dot(xx[1:,1:].dot((xx[1:,1:]/(E[1:].conjugate() - ein[1] + ein_sigma)).dot(xx[1:,0]/(E[1:] - ein[1]))))
    + (xx[0,1:] / (E[1:].conjugate() + ein[2])).dot(xx[1:,1:].dot((xx[1:,1:]/(E[1:].conjugate() - ein[0] + ein_sigma)).dot(xx[1:,0]/(E[1:] - ein[0]))))
    + (xx[0,1:] / (E[1:].conjugate() + ein[1])).dot(xx[1:,1:].dot((xx[1:,1:]/(E[1:].conjugate() - ein[0] + ein_sigma)).dot(xx[1:,0]/(E[1:] - ein[0]))))
    + (xx[0,1:] / (E[1:].conjugate() + ein[1])).dot(xx[1:,1:].dot((xx[1:,1:]/(E[1:].conjugate() - ein[2] + ein_sigma)).dot(xx[1:,0]/(E[1:] - ein[2]))))
    + (xx[0,1:] / (E[1:].conjugate() + ein[0])).dot(xx[1:,1:].dot((xx[1:,1:]/(E[1:] - ein[1] - ein[2])).dot(xx[1:,0]/(E[1:] - ein[1]))))
    + (xx[0,1:] / (E[1:].conjugate() + ein[0])).dot(xx[1:,1:].dot((xx[1:,1:]/(E[1:] - ein[2] - ein[1])).dot(xx[1:,0]/(E[1:] - ein[2]))))
    + (xx[0,1:] / (E[1:].conjugate() + ein[1])).dot(xx[1:,1:].dot((xx[1:,1:]/(E[1:] - ein[0] - ein[2])).dot(xx[1:,0]/(E[1:] - ein[0]))))
    + (xx[0,1:] / (E[1:].conjugate() + ein[1])).dot(xx[1:,1:].dot((xx[1:,1:]/(E[1:] - ein[2] - ein[0])).dot(xx[1:,0]/(E[1:] - ein[2]))))
    + (xx[0,1:] / (E[1:].conjugate() + ein[2])).dot(xx[1:,1:].dot((xx[1:,1:]/(E[1:] - ein[0] - ein[1])).dot(xx[1:,0]/(E[1:] - ein[0]))))
    + (xx[0,1:] / (E[1:].conjugate() + ein[2])).dot(xx[1:,1:].dot((xx[1:,1:]/(E[1:] - ein[1] - ein[0])).dot(xx[1:,0]/(E[1:] - ein[1]))))
    + (xx[0,1:] / (E[1:].conjugate() + ein[0])).dot(xx[1:,1:].dot((xx[1:,1:]/(E[1:].conjugate() + ein_sigma - ein[2])).dot(xx[1:,0]/(E[1:].conjugate() + ein_sigma))))
    + (xx[0,1:] / (E[1:].conjugate() + ein[0])).dot(xx[1:,1:].dot((xx[1:,1:]/(E[1:].conjugate() + ein_sigma - ein[1])).dot(xx[1:,0]/(E[1:].conjugate() + ein_sigma))))
    + (xx[0,1:] / (E[1:].conjugate() + ein[2])).dot(xx[1:,1:].dot((xx[1:,1:]/(E[1:].conjugate() + ein_sigma - ein[0])).dot(xx[1:,0]/(E[1:].conjugate() + ein_sigma))))
    + (xx[0,1:] / (E[1:].conjugate() + ein[2])).dot(xx[1:,1:].dot((xx[1:,1:]/(E[1:].conjugate() + ein_sigma - ein[1])).dot(xx[1:,0]/(E[1:].conjugate() + ein_sigma))))
    + (xx[0,1:] / (E[1:].conjugate() + ein[1])).dot(xx[1:,1:].dot((xx[1:,1:]/(E[1:].conjugate() + ein_sigma - ein[2])).dot(xx[1:,0]/(E[1:].conjugate() + ein_sigma))))
    + (xx[0,1:] / (E[1:].conjugate() + ein[1])).dot(xx[1:,1:].dot((xx[1:,1:]/(E[1:].conjugate() + ein_sigma - ein[0])).dot(xx[1:,0]/(E[1:].conjugate() + ein_sigma))))
    )
    
    return gamma

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

