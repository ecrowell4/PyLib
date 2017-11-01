# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 11:21:17 2017

@author: Ethan Crowell
"""
def beta_tst(E, xx, ein=[0,0]):
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
    import numpy as np
    e=1

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
            
    #Calculate beta
    beta = 0.5 * e**2 * ((xx[0,1:]/(E[1:] - ein[0] - ein[1])).dot(xx[1:,1:].dot(xx[1:,0]/(E[1:] - ein[0])))
    + (xx[0,1:]/(E[1:] - ein[1] - ein[0])).dot(xx[1:,1:].dot(xx[1:,0]/(E[1:] - ein[1])))
    + (xx[0,1:]/(E[1:] - ein[1])).dot(xx[1:,1:].dot(xx[1:,0]/(E[1:].conjugate() + ein[0])))
    + (xx[0,1:]/(E[1:] - ein[0])).dot(xx[1:,1:].dot(xx[1:,0]/(E[1:].conjugate() + ein[1])))
    + (xx[0,1:]/(E[1:].conjugate() + ein[0] + ein[1])).dot(xx[1:,1:].dot(xx[1:,0]/(E[1:].conjugate() + ein[0])))
    + (xx[0,1:]/(E[1:].conjugate() + ein[1] + ein[0])).dot(xx[1:,1:].dot(xx[1:,0]/(E[1:].conjugate() + ein[1]))))
            
    return beta

def beta_eee(E, xx, ein=[0,0]):
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
    import numpy as np
    e=1

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
            
    #Calculate beta
    beta = 0.5 * e**2 * ((xx[0,1:]/(E[1:] - ein[0] - ein[1])).dot(xx[1:,1:].dot(xx[1:,0]/(E[1:] - ein[0])))
    + (xx[0,1:]/(E[1:] - ein[1] - ein[0])).dot(xx[1:,1:].dot(xx[1:,0]/(E[1:] - ein[1])))
    + (xx[0,1:]/(E[1:] - ein[1])).dot(xx[1:,1:].dot(xx[1:,0]/(E[1:].conjugate() + ein[0])))
    + (xx[0,1:]/(E[1:] - ein[0])).dot(xx[1:,1:].dot(xx[1:,0]/(E[1:].conjugate() + ein[1])))
    + (xx[0,1:]/(E[1:].conjugate() + ein[0] + ein[1])).dot(xx[1:,1:].dot(xx[1:,0]/(E[1:].conjugate() + ein[0])))
    + (xx[0,1:]/(E[1:].conjugate() + ein[1] + ein[0])).dot(xx[1:,1:].dot(xx[1:,0]/(E[1:].conjugate() + ein[1]))))
            
    return beta