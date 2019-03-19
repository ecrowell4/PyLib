# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 11:21:17 2017

@author: Ethan Crowell
"""

import numpy as np
from nlopy.hyperpol.sum_over_states import sos_utils


def beta_eee(E, xx, units, omega=[0,0], intrinsic=False, n=0):
    """Returns the diagonal element of the first hyperpolarizability.

    Input
        E : np.array
            array of ordered eigenenergies. Damping is included by letting this array
            have complex entries.
        xx : np.array
            transition matrix
        omega : list
            incident field frequencies [omega1, omega2]
        intrinsic : bool
            if True, then xx must be normalized by xmax before input
            and E must be normalized by E[1] before input.
        n : int
            state system is assumed to be in (i.e. n=0 -> ground state)

    Output
        beta : complex
            the first hyperpolarizability.
    """

    # assert consistent dimensions
    assert len(E)==len(xx[0]), "dimensions of E and xx do not match."

    # determine number of eigenstates to be used in computing beta
    num_states = len(E)

    #Take all mu -> bar{mu}
    xx = xx - xx[0,0] * np.eye(num_states)

    #Take all Em -> Emn
    E = E - E[n]
            
    #Calculate beta: note that factor of 0.5 comes from the average over permutations.
    beta = 0.5 * units.e**3 * (
        (np.delete(xx[n,:], n) * sos_utils.D2(np.delete(E, n), omega[0], omega[1], units)).dot(np.delete(np.delete(xx, n, 0), n, 1).dot(np.delete(xx[:,n], n) 
    * sos_utils.D1(np.delete(E, n), omega[0], units)))
    + (np.delete(xx[n,:], n) * sos_utils.D2(np.delete(E, n), omega[1], omega[0], units)).dot(np.delete(np.delete(xx, n, 0), n, 1).dot(np.delete(xx[:,n], n) * sos_utils.D1(np.delete(E, n), omega[1], units)))
    + (np.delete(xx[n,:], n) * sos_utils.D1(np.delete(E, n), omega[1], units)).dot(np.delete(np.delete(xx, n, 0), n, 1).dot(xx[1:,0] * sos_utils.D1(np.delete(E, n).conjugate(), -omega[0], units)))
    + (np.delete(xx[n,:], n) * sos_utils.D1(np.delete(E, n), omega[0], units)).dot(np.delete(np.delete(xx, n, 0), n, 1).dot(xx[1:,0] * sos_utils.D1(np.delete(E, n).conjugate(), -omega[1], units)))
    + (np.delete(xx[n,:], n) * sos_utils.D2(np.delete(E, n).conjugate(), -omega[0], -omega[1], units)).dot(np.delete(np.delete(xx, n, 0), n, 1).dot(xx[1:,0] * sos_utils.D1(np.delete(E, n).conjugate(), -omega[0], units)))
    + (np.delete(xx[n,:], n) * sos_utils.D2(np.delete(E, n).conjugate(), -omega[1], -omega[0], units)).dot(np.delete(np.delete(xx, n, 0), n, 1).dot(xx[1:,0] * sos_utils.D1(np.delete(E, n).conjugate(), -omega[1], units)))
    )
    
    if intrinsic is True:
        # normalize by constant (this assumes E and xx have been entered as E/E[1] and xx/xmax)
        return beta * (3/4)**(3/4) / 3
    else:
        # return the actual value of beta.
        return beta

def betaint(e, xi, ijk = [0,0,0], ein = [0,0], Ne = 1, start = 0):
    """Calculates the first hyperpolarizability, beta, as a function of two 
    input photon energies and complex molecular transition and energy 
    information. The input information must first be normalized to x_max and 
    E_10. Calculates the one tensor component specified.
    Input: 
        xi = [ mu_x, mu_y, mu_z ]/x_max the dipole moment matrices in cartesian
            coordinates normalized to the maximum transition strength x_10(max)
            where mu_x = [<i|e x|j>]
        e = [E_nm - i/2 Gamma_nm]/E_10 the electronic eigen frequencies 
            corresponding to the transitions from the ith eigen state to the
            ground state normailized to E_10
            OR
            [E_n0]/E_10 as a vector
        *The sizes of xi and E represent the number of states considered in
            this calculation and therefore must be consistant.*
        ijk = [0,1,2] the tensor components of beta where x=0, y=1, z=2
    Option:
        Calculate dispersion:
        ein = hbar*[w_j, w_k]/E_10 input photon energies, which implies output 
            energy of (ein[0] + ein[1])*E_10
        Start and end in a given quantum state
        start = int some integer state which is represented by the xi and E info
    Output:
        betaint = beta_ijk(-w_sigma; w[0], w[1])/betamax
    """
    import numpy as np
    
    #Check that start is an integer
    if start != abs(int(start)):
        print("State number must be an integer.")
        return 0
    
    #Determine dimensionality allowed by the supplied information
    if len(np.shape(xi)) == 2:
        xi = np.array([xi])
    else:
        xi = np.array(xi)
    if len(xi) < np.max(ijk):
        print("Dimensionality error between transitions and tensor component")
        return 0    
        
    #Find the number of electronic states supplied
    NumStates = len(xi[0])
    
    #Check for consistancy between transitions and energies
    if NumStates != len(e):
        print('Err: Inconsistant electronic state information.')
        return 0
    
    #Check if supplied with matrix E_nm or vector E_n0
    if len(np.shape(e)) == 1:
        e = np.array(list([np.outer(e, np.ones(NumStates)) - np.outer(np.ones(NumStates), e)])*(max(ijk)+1))

    #Take all mu -> bar{mu}
    xi = xi - np.array([xi[:, start, start][i]*np.eye(NumStates) for i in range(len(xi))])
    
    #Calculate beta
    betaint = Ne**(-1.5) * 1/6.*(0.75)**0.75*( np.dot( np.dot( np.delete(xi[ijk[0], start, :], start),
                np.delete(np.delete( xi[ijk[1]]/(e[ijk[1]] + np.outer(np.ones(NumStates), e[ijk[2], :, start]) - ein[0] - ein[1]), start, 0), start, 1) ),
                np.delete(xi[ijk[2], :, start]/(e[ijk[2], :, start] - ein[1]), start) ) +
            np.dot( np.dot( np.delete(xi[ijk[1], start, :]/(np.conjugate(e[ijk[1], :, start]) + ein[1]), start) ,
                np.delete( np.delete(xi[ijk[0]], start, 0), start, 1) ),            
                np.delete( xi[ijk[2], :, start]/(e[ijk[2], :, start] - ein[0]), start) )+
            np.dot( np.dot( np.delete( xi[ijk[2], start, :]/(np.conjugate(e[ijk[2], :, start]) + ein[1]), start) ,
                np.delete( np.delete(xi[ijk[0], :, :], start, 0), start, 1) ),
                np.delete( xi[ijk[1], :, start]/(e[ijk[1], :, start] - ein[0]), start) )+
            np.dot( np.dot( np.delete( xi[ijk[0], start, :], start),
                np.delete( np.delete( xi[ijk[2], :, :]/(e[ijk[2], :, :] + np.outer(np.ones(NumStates), e[ijk[1], :, start]) - ein[0] - ein[1]), start, 0), start, 1) ),
                np.delete( xi[ijk[1], :, start]/(e[ijk[1], :, start] - ein[0]), start) )+
            np.dot( np.dot( np.delete(xi[ijk[1], start, :]/(np.conjugate(e[ijk[1], :, start]) + ein[0]), start) ,
                np.delete( np.delete(xi[ijk[2], :, :], start, 0), start, 1) ),
                np.delete( xi[ijk[0], :, start]/(np.conjugate(e[ijk[2], :, start]) + ein[0] + ein[1]), start) )+
            np.dot( np.dot( np.delete( xi[ijk[2], start, :]/(np.conjugate(e[ijk[2], :, start]) + ein[1]), start) ,
                np.delete( np.delete( xi[ijk[1], :, :], start, 0), start, 1) ),
                np.delete( xi[ijk[0], :, start]/(np.conjugate(e[ijk[1], :, start]) + ein[0] + ein[1]), start) ) 
            )
            
    return betaint
    
def betaintterms(E, xi, ijk = [0,0,0], ein = [0,0], Ne = 1, start = 0):
    """Calculates the first hyperpolarizability, beta, as a function of two 
    input photon energies and complex molecular transition and energy 
    information. The input information must first be normalized to x_max and 
    E_10.
    Input: 
        xi = [ mu_x, mu_y, mu_z ]/x_max the dipole moment matrices in cartesian
            coordinates normalized to the maximum transition strength x_10(max)
            where mu_x = [<i|e x|j>]
        E = [E_nm - i/2 Gamma_nm]/E_10 the electronic eigen frequencies 
            corresponding to the transitions from the ith eigen state to the
            ground state normailized to E_10
            OR
            [E_n0]/E_10 as a vector
        *The sizes of xi and E represent the number of states considered in
            this calculation and therefore must be consistant.*
        ijk = [0,1,2] the tensor components of beta where x=0, y=1, z=2
    Option:
        Calculate dispersion:
        ein = hbar*[w_j, w_k]/E_10 input photon energies, which implies output 
            energy of (ein[0] + ein[1])*E_10
        Start and end in a given quantum state
        start = int some integer state which is represented by the xi and E info
    Output:
        betaint = beta_ijk(-w_sigma; w[0], w[1])/betamax
    """
    import numpy as np
    
    #Check that start is an integer
    if start != abs(int(start)):
        print("State number must be an integer.")
        return 0
    
    #Determine dimensionality allowed by the supplied information
    if len(np.shape(xi)) == 2:
        xi = np.array([xi])
    else:
        xi = np.array(xi)
    if len(xi) < np.max(ijk):
        print("Dimensionality error between transitions and tensor component")
        return 0    
        
    #Find the number of electronic states supplied
    NumStates = len(xi[0])
    
    #Check for consistancy between transitions and energies
    if NumStates != len(E):
        print('Err: Inconsistant electronic state information.')
        return 0
    
    #Check if supplied with matrix E_nm or vector E_n0
    if len(np.shape(E)) == 1:
        E = np.array(list([np.outer(E, np.ones(NumStates)) - np.outer(np.ones(NumStates), E)])*(max(ijk)+1))

    #Take all mu -> bar{mu}
    xi = xi - np.array([xi[:, start, start][i]*np.eye(NumStates) for i in range(len(xi))])

def betatensor(E, xi, ein = [0,0], Ne = 1, start = 0):
    """Iterates over the possible tensor components of beta to obtain the full
    tensor."""
    
    import numpy as np
    
    #Check that start is an integer
    if start != abs(int(start)):
        print("State number must be an integer.")
        return 0
    
    #Determine dimensionality allowed by the supplied information
    if len(np.shape(xi)) == 2:
        xi = np.array([xi])
    else:
        xi = np.array(xi)
    
    #Calculate tensor components of beta
    betaALL = np.array([[[betaint(E, xi, ijk = [i,j,k], ein = ein, Ne = Ne, start = start)
                    for i in range(len(xi))] for j in range(len(xi))] for k in range(len(xi))])
    return betaALL
    
def betatensormax(E, xi, ein = [0,0], Ne = 1, start = 0):
    """Rotates the full beta tensor to maximize a single diagonal component. 
    Currently only works for 2D structures, this needs to be fixed."""
    
    import numpy as np
    
    betaALL = betatensor(E, xi, ein = ein, Ne = Ne, start = start)
    
    theta = np.linspace(0, 2*np.pi, num=100)
    betaxxx = (np.cos(theta)**3 * betaALL[0,0,0] + 3*np.cos(theta)**2 * np.sin(theta) * betaALL[0,0,1] +
               3*np.cos(theta)*np.sin(theta)**2 * betaALL[0,1,1] + np.sin(theta)**3 * betaALL[1,1,1] )
               
    return max(betaxxx), theta[np.argmax(betaxxx)]

def betadeletedstate(xi, e, delstate, ijk=[0,0,0], start=0):
    import numpy as np
    
    if delstate == start:
        print("Can't ommit start state.")
        return 0
        
    betatotal = betaint(xi, e, ijk=ijk, start=start)
    
    betamissing = (betatotal - betaint(
            np.delete(np.delete(xi, delstate, 0), delstate, 1), 
            np.delete(e, delstate) , ijk=ijk, start=start))/betatotal
        
    return betamissing