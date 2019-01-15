import numpy as np
import copy
import sys

# shorthand definition for np.delete
Del = np.delete

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
        
    Output
    	D1 : np.array
    		propogator for first order process
    """
    
    # Compute and return the propogator
    return 1 / (E - units.hbar * omega)

def D2(E, omega1, omega2, units):
    """Returns the propogator for a second order process.

    Input
        E : np.array
            array of eigenenergies
        omega : np.array
            [omega1, omega2] array of incident field frequencies
        units : class
            class whose attributes are the fundamental constants

    Output
        D2 : np.array
            propogator for second order process.
    """

    # Compute and return the propogator
    return 1 / (E - units.hbar * omega1 - units.hbar * omega2)

def D3(E, omega1, omega2, omega3, units):
    """Returns the propogator for a third order process.

    Input
        E : np.array
            array of eigenenergies
        omega : float
            incident field intensities
        units : class
            class whose attributes are the fundamental constants
    
    Output
        D3 : np.array
            propogator for third order process.
    """

    # Compute and return the propogator
    return 1 / (E - units.hbar * omega1 - units.hbar * omega2 - units.hbar * omega3)




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


   
def gamma_term11(xx, E, omega1, omega2, omega3, units, n=0):
    """Returns the first term of the first summand in SOS expression
    for gamma_eee, as written [FILL IN LOCATION]"""
    term11 = units.e**4 * Del(xx[n,:], n).dot(
        (Del(Del(xx, n, 0), n, 1) * D3(Del(E, n), omega1, omega2, omega3, units)).dot(
            (Del(Del(xx, n, 0), n, 1) * D2(Del(E, n), omega1, omega2, units)).dot(
                (Del(xx[:,n], n) * D1(Del(E, n), omega1, units)))))
    
    return term11

def gamma_term12(xx, E, omega1, omega2, omega3, units, n=0):
    """Returns the second term of the first summand in SOS expression
    for gamma_eee, as written [FILL IN LOCATION]"""
    term12 = units.e**4 * Del(xx[n,:], n).dot(
        (Del(Del(xx, n, 0), n, 1) * D1(Del(E.conjugate(), n), -omega1, units)).dot(
            (Del(Del(xx, n, 0), n, 1) * D2(Del(E, n), omega3, omega2, units)).dot(
                (Del(xx[:,n], n) * D1(Del(E, n), omega1, units)))))
    
    return term12

def gamma_term13(xx, E, omega1, omega2, omega3, units, n=0):
    """Returns the thrid term of the first summand in SOS expression
    for gamma_eee, as written [FILL IN LOCATION]"""
    term13 =  units.e**4 * Del(xx[n,:], n).dot(
        (Del(Del(xx, n, 0), n, 1) * D1(Del(E, n), omega1, units)).dot(
            (Del(Del(xx, n, 0), n, 1) * D2(Del(E.conjugate(), n), -omega3, -omega2, units)).dot(
                (Del(xx[:,n], n) * D1(Del(E.conjugate(), n), -omega3, units)))))
    
    return term13

def gamma_term14(xx, E, omega1, omega2, omega3, units, n=0):
    """Returns the fourth term of the first summand in SOS expression
    for gamma_eee, as written [FILL IN LOCATION]"""
    term14 = units.e**4 * Del(xx[n,:], n).dot(
        (Del(Del(xx, n, 0), n, 1) * D3(Del(E.conjugate(), n), -omega1, -omega2, -omega3, units)).dot(
            (Del(Del(xx, n, 0), n, 1) * D2(Del(E.conjugate(), n), -omega1, -omega2, units)).dot(
                (Del(xx[:,n], n) * D1(Del(E.conjugate(), n), -omega2, units)))))
    
    return term14

def gamma_term21(xx, E, omega1, omega2, omega3, units, n=0):
    """Returns the first term of the second summand in sos expression
    for gamma_eeee"""
    term21 = units.e**4 * ((Del(xx[n,:], n) * D3(Del(E, n), omega1, omega2, omega3, units)).dot(
        (Del(xx[:,n], n) * D1(Del(E, n), omega3, units)))) * Del(xx[n,:], n).dot(
    Del(xx[:,n], n) * D1(Del(E, n), omega3))

    return term21

def gamma_term22(xx, E, omega1, omega2, omega3, units, n=0):
    """Returns the second term of the second summand in sos expression
    for gamma_eeee"""
    term22 = units.e**4 * ((Del(xx[n,:], n) * D1(Del(E.conjugate(), n), -omega2, units)).dot(
        (Del(xx[:,n], n) * D1(Del(E, n), omega1, units)))) * Del(xx[n,:], n).dot(
    Del(xx[:,n], n) * D1(Del(E, n), omega3))

    return term22

def gamma_term23(xx, E, omega1, omega2, omega3, units, n=0):
    """Returns the third term of the second summand in sos expression
    for gamma_eeee"""
    term23 = units.e**4 * ((Del(xx[n,:], n) * D3(Del(E.conjugate(), n), -omega1,-omega2, -omega3, units)).dot(
        (Del(xx[:,n], n) * D1(Del(E.conjugate(), n), -omega3, units)))) * Del(xx[n,:], n).dot(
    Del(xx[:,n], n) * D1(Del(E.conjguate(), n), -omega1))

    return term23

def gamma_term24(xx, E, omega1, omega2, omega3, units, n=0):
    """Returns the fourth term of the second summand in sos expression
    for gamma_eeee"""
    term24 = units.e**4 * ((Del(xx[n,:], n) * D1(Del(E.conjugate(), n), -omega1, units)).dot(
        (Del(xx[:,n], n) * D1(Del(E, n), omega2, units)))) * Del(xx[n,:], n).dot(
    Del(xx[:,n], n) * D1(Del(E.conjugate(), n), -omega3))

    return term24

def gamma_term31(xx, E, omega1, omega2, omega3, units, n=0):
    """Returns the first term of the third summand in sos expression
    for gamma_eeee"""
    term31 = units.e**4 * (Del(xx[n,:], n) * D3(Del(E, n), omega1, omega2, omega3, units)).dot(
        (Del(xx[:,n], n) * D2(Del(E, n), omega1, omega2, units))
        ) / omega1 / omega2

    return term31

def gamma_term32(xx, E, omega1, omega2, omega3, units, n=0):
    """Returns the second term of the third summand in sos expression
    for gamma_eeee"""
    term32 = units.e**4 * (Del(xx[n,:], n) * D3(Del(E.conjugate(), n), -omega1, -omega2, -omega3, units)).dot(
        (Del(xx[:,n], n) * D2(Del(E.conjugate(), n), -omega1, -omega2, units))
        ) / omega1 / omega2

    return term32

def gamma_term33(xx, E, omega1, omega2, omega3, units, n=0):
    """Returns the first term of the third summand in sos expression
    for gamma_eeee"""
    term33 = units.e**4 * (Del(xx[n,:], n) * D1(Del(E.conjugate(), n), -omega1, units)).dot(
        (Del(xx[:,n], n) * D2(Del(E, n), omega2, omega3, units))
        ) / omega1 / omega2

    return term33

def gamma_term34(xx, E, omega1, omega2, omega3, units, n=0):
    """Returns the first term of the third summand in sos expression
    for gamma_eeee"""
    term34 = units.e**4 * (Del(xx[n,:], n) * D1(Del(E, n), omega1, units)).dot(
        (Del(xx[:,n], n) * D2(Del(E.conjugate(), n), -omega2, -omega3, units))
        ) / omega1 / omega2

    return term34