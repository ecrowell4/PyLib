# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 12:38:46 2017

@author: Ethan Crowell
"""

import numpy as np
from nlopy.hyperpol.sum_over_states import sos_utils

def alpha_ee(E, xx, units, omega=0, intrinsic=False, n=0):
    """Calculates the polarizability alpha for a system in the ground
    
    Input
        E : np.array(N)
            energy spectrum
        xx : np.array((N,N))
            transition matrix
        units : class
            class object containing fundamental constants
        omega : float
            frequency of incident electric field
        n : int
            state system is assumes to be in (i.e. n=0 -> ground state)
            
        
    Output
        alpha : complex
            electric polarizability
    """

    # assert consistent dimensions
    assert len(E)==len(xx[0]), "dimensions of E and xx do not match."
    
    # Take all Em -> Emn
    E = E - E[n]
    
    alpha =  units.e**2 * (np.delete(xx[n,:], n).dot(np.delete(xx[:,n], n) * sos_utils.D1(np.delete(E, n), omega, units))
                           + np.delete(xx[n,:], n).dot(np.delete(xx[:,n], n) * sos_utils.D1(np.delete(E.conjugate(), n), -omega, units))
                          )

    if intrinsic is True:
        return alpha / 2
    else:
        return alpha


def alpha_mm(E, L, I, units, omega=0, canonical=False):
    """Returns the canonical, linear, magnetic polarizability alpha at input frequency
    omega:
        m = alpha * B(oemga)
    
    INPUT
        E : np.array
            eigenenergies
        L : np.array
            angular momentum matrix
        I : np.complex
            ground state expectation value of moment of inertia operator: <0|I|0>
        units : Class
            class object containing fundamental constants
        omega : float
            frequency of incident magnetic perturbation
        canonical : bool
            if True, return alpha using canonical angular momentum, not mechanical
        
    OUTPUT
        alpha : complex
            magnetic polarizability
    """
    
        
    # Evaluate the SOS term in equation for alpha^{mm}
    alpha = units.g**2 * (L[0,1:].dot((L[1:,0] * sos_utils(np.delete(E, 1), omega, units)))
                       + L[0,1:].dot((L[1:,0] * sos_utils(np.delete(E.conjugate(), 1), omega, units))))
    
    # Include the diamagnetic term (i.e. Faraday term) if the user doesn't
    # specify not to
    if canonical is True:
        return alpha
    else:
        return alpha - g**2 * I


#==============================================================================
# The following functions were written for use in quasi-degnerate systems.
# However, I don't think they are pertinent to anything we're doing.
    
def alpha_ee_DontUse(E, XX, units, omega=0):   # I believe this function is out of date
    """Calculates the polarizability alpha as a function of the energy spectrum
    and transition matrix. The SOS expression derived from time-dependent
    perturbation theory is evaluated using the first num_states states.
    
    Input
        E : np.array(N)
            energy spectrum
        XX : list [xx_exp, xx_pert]
            List whose elements are the transition matrices that enter into SOS
            due to expection value and perturbation, respectively. These are 
            typically the same except for when quasi degenerate methods are
            implemented.
        units : class
            class whose attributes are the fundamental constants hbar, e, m, c, etc.
        omega : float, default value of zero
            frequency of incident electric field
        
    Output
        alpha : complex
            polarizability describing the linear response of the polarization
            to an electric field.
    """
    
    # Determine number of states in input:
    num_states = len(E)
    
    # We assume that transition moments that are np.allclose to zero actually vanish,
    #   and are only nonzero due to numerical errors:
    if np.allclose(XX[1][1,0], 0):
        start = 2
    else:
        start = 1
            
    return units.e**2 * (XX[0][0,start:].dot(XX[1][start:,0] * sos.D1(E, omega, units, start)) 
        + XX[1][0,start:].dot(XX[0][start:,0] * sos.D1(E.conjugate(), -omega, units, start)))
    
def alpha_quasi_degen(E_prime, xx_prime, E10, coeff_diff, units):
    """Returns the linear polarizability for a system in the primed basis. In this
    basis, polarizations of all orders have terms that are linear in the applied
    field, so we take these terms into account.
    
    Input
        E_prime : np.array
            modified eigenergies, first two states are degenerate
        xx_prime : np.array
            position matrix in primed basis
        E10 : float
            first ground excited state energy difference, unprimed basis
        coeff_diff : float
            |alpha_0|^2 - |beta_0|^2, where |0'> = alpha_0 |0> + beta_0 |1>
        units : class
            fundamental constants
            
    Output
        alpha : complex
            linear polarizability
    """
    
    E_denom = 2*(E_prime[2:] - E_prime[0]) -  E10 * coeff_diff   
    return 4 * (xx_prime[0,2:].dot(xx_prime[2:,0] / E_denom)
                + xx_prime[1,2:].dot(xx_prime[2:,1] / E_denom))
    
def alpha_quasi_degen2(E_prime, E10, xx_prime, xi, units, omega=0):
    """Returns the linear polarizability for a system in the primed basis. In this
    basis, polarizations of all orders have terms that are linear in the applied
    field, so we take these terms into account.
    
    Input
        E_prime : np.array
            modified eigenergies, first two states are degenerate
        xx_prime : np.array
            position matrix in primed basis
        E10 : float
            first ground excited state energy difference, unprimed basis
        coeff_diff : float
            |alpha_0|^2 - |beta_0|^2, where |0'> = alpha_0 |0> + beta_0 |1>
        units : class
            fundamental constants
            
    Output
        alpha : complex
            linear polarizability
    """
    
    alpha = units.e**2 * (sos.alpha_term(E_prime, E10, xx_prime, [0,0], xi, omega)
                               + sos.alpha_term(E_prime, E10, xx_prime, [1,1], xi, omega)
                               + sos.alpha_term(E_prime, E10, xx_prime, [0,1], xi, omega)
                               + sos.alpha_term(E_prime, E10, xx_prime, [1,0], xi, omega))
    
    return alpha
#==============================================================================
#     return units.e**2 * (xi[0].conjugate() 
#                          * (xi[0] * xx_prime[0,2:].dot(xx_prime[2:,0] / (E_prime[2:]-E_prime[0] - omega))
#                          + xi[1] * xx_prime[0,2:].dot(xx_prime[2:,1] / (E_prime[2:]-E_prime[0]-omega)))
#                          + xi[0] 
#                          * (xi[0].conjugate() * xx_prime[0,2:].dot(xx_prime[2:,0] / (E_prime[2:]-E_prime[0]+omega))
#                          + xi[1].conjugate() * xx_prime[1,2:].dot(xx_prime[2:,0] / (E_prime[2:]-E_prime[0]+omega)))
#                          + xi[1].conjugate() 
#                          * (xi[0] * xx_prime[1,2:].dot(xx_prime[2:,0] / (E_prime[2:]-E_prime[0] - omega))
#                          + xi[1] * xx_prime[1,2:].dot(xx_prime[2:,1] / (E_prime[2:]-E_prime[0]-omega)))
#                          + xi[1] 
#                          * (xi[0].conjugate() * xx_prime[0,2:].dot(xx_prime[2:,1] / (E_prime[2:]-E_prime[0] + omega))
#                          + xi[1].conjugate() * xx_prime[1,2:].dot(xx_prime[2:,1] / (E_prime[2:]-E_prime[0] + omega)))
#                          )
#==============================================================================
