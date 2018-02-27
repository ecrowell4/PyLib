import numpy as np
import copy
import sys

def D1(E, omega, units):
    """Returns the propogator for a first order process.
    
    Input
    	E : np.array
        	array of eigenergies.
        	Note: if omega is take as negative, then user 
        	should pass E.conjugate() instead of E.
    omega : float
        input frequency
    units : class
        class whose attributes are the fundamental constants hbar, e, m, c, etc
    start : int
        first state included in SOS expression
        
    Output
    	D1 : np.array
    		propogator for first order process
    """
    
    # Compute and return the propogator
    return 1 / (E - units.hbar*omega)




def damping_coeffs(E, xx, units):
    """Returns and array with the damping terms derived from Fermi's
    Golden Rule (see M Kuzyk, J Chem Phys 125, 2006).

    Input
        E : np.array
            Eigenenergies
        xx : np.array
            Position matrix
        units : class
            Class whose attributes are the fundamental constants

    Output
        Gamma : np.array
            Array with same shape as E representing the damping term for each state.
    """

    Gamma =  (2 / 3) * ((E - E[0]) / units.hbar / units.c)**3 * units.e**2 * abs(xx[0])**2

    return Gamma

def alpha_term(E, delta, xx, ij, xi, omega):
    """Returns an individual alpha_n from the sum alpha = Sum_n[alpha_n].
    
    Input
        E_prime : np.array
            modified eigenstates
        E10 : float
            energy difference between first excited and ground state
        xx_prime : np.array
            modified position matrix
        [i,j] : list
            start and end indices for the term, must be either 0 or 1.
        xi : array
            coefficients that diagonalize the modified perturbation
        omega : float
            input frequency
        units : class
            fundamental constants
            
    Output
        alpha_n : complex
            individual term from SOS expression for alpha
    """
    
    if ij[0] == 0:
        sgn1 = 1
    elif ij[0] == 1:
        sgn1 = -1
        
    if ij[1] == ij[1]:
        sgn2 = sgn1
    else:
        sgn2 = -sgn1
        
    return 4 * xi[ij[0]].conjugate() * xi[ij[1]] * (xx[ij[0],2:].dot(xx[2:,ij[1]] / (2*(E[2:] - E[0]) + sgn1*delta))
             + xx[ij[0],2:].dot(xx[2:,ij[1]] / (2*(E[2:] - E[0]) + sgn2*delta)))


    
    
    