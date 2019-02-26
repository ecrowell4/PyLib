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


   
def gamma_term11(L, I, E, omega1, omega2, omega3, units, n=0):
    """Returns the first term of the first summand in SOS expression
    for gamma_mmmm, as written [FILL IN LOCATION]

    Input
        xx : np.array
            transition matrix
        E : np.array
            complex eigenenergies (imaginary part is due to damping)
        omega1-3 : np.float
            incident field frequencies
        units : class
            fundamental constants

    Output
        gamma_term : complex
            the first sum in the first set of terms of gamma_mmmm
    """

    term11 = units.g**4 * (Del(L[n,:], n) * D3(Del(E,n), omega1, omega2, omega3, units)).dot(
        (Del(Del(L, n, 0), n, 1) * D2(Del(E, n), omega1, omega2, units)).dot(
            Del(Del(L, n, 0), n, 1).dot(
                (Del(L[:,n], n) * D1(Del(E, n), omega1, units)))))
    
    return term11

def gamma_term12(L, I, E, omega1, omega2, omega3, units, n=0):
    """Returns the second term of the first summand in SOS expression
    for gamma_eee, as written [FILL IN LOCATION]

    Input
        xx : np.array
            transition matrix
        E : np.array
            complex eigenenergies (imaginary part is due to damping)
        omega1-3 : np.float
            incident field frequencies
        units : class
            fundamental constants

    Output
        gamma_term : complex
            the second sum in the first set of terms of gamma_eeee
    """

    term12 = units.g**4 * (Del(L[n,:], n) * D1(Del(E.conjugate(),n), -omega1, units)).dot(
        (Del(Del(L, n, 0), n, 1) * D2(Del(E, n), omega2, omega3, units)).dot(
            Del(Del(L, n, 0), n, 1).dot(
                (Del(L[:,n], n) * D1(Del(E, n), omega3, units))
                )
            )
        )
    
    return term12

def gamma_term13(L, I, E, omega1, omega2, omega3, units, n=0):
    """Returns the thrid term of the first summand in SOS expression
    for gamma_eee, as written [FILL IN LOCATION]

    Input
        xx : np.array
            transition matrix
        E : np.array
            complex eigenenergies (imaginary part is due to damping)
        omega1-3 : np.float
            incident field frequencies
        units : class
            fundamental constants

    Output
        gamma_term : complex
            the third sum in the first set of terms of gamma_eeee
    """

    term13 =  units.g**4 * (Del(L[n,:], n) * D1(Del(E.conjugate(),n), -omega3, units)).dot(
        (Del(Del(L, n, 0), n, 1) * D2(Del(E.conjugate(), n), -omega3, -omega2, units)).dot(
            Del(Del(L, n, 0), n, 1).dot(
                (Del(L[:,n], n) * D1(Del(E, n), omega1, units))
                )
            )
        )
    
    return term13

def gamma_term14(L, I, E, omega1, omega2, omega3, units, n=0):
    """Returns the fourth term of the first summand in SOS expression
    for gamma_eee, as written [FILL IN LOCATION]

    Input
        xx : np.array
            transition matrix
        E : np.array
            complex eigenenergies (imaginary part is due to damping)
        omega1-3 : np.float
            incident field frequencies
        units : class
            fundamental constants

    Output
        gamma_term : complex
            the fourth sum in the first set of terms of gamma_eeee
    """

    term14 = units.g**4 *(Del(L[n,:], n) * D1(Del(E.conjugate(), n), -omega2, units)).dot(
        (Del(Del(L, n, 0), n, 1) * D2(Del(E.conjugate(), n), -omega1, -omega2, units)).dot(
            Del(Del(L, n, 0), n, 1).dot(
                (Del(L[:,n], n) * D3(Del(E.conjugate(), n), -omega1, -omega2, -omega3, units))
                )
            )
        )
    
    return term14

def gamma_term21(L, I, E, omega1, omega2, omega3, units, n=0):
    """Returns the first term of the second summand in sos expression
    for gamma_eeee

    Input
        xx : np.array
            transition matrix
        E : np.array
            complex eigenenergies (imaginary part is due to damping)
        omega1-3 : np.float
            incident field frequencies
        units : class
            fundamental constants

    Output
        gamma_term : complex
            the first sum in the second set of terms of gamma_eeee
    """

    term21 = units.g**4 * ((Del(L[n,:], n) * D3(Del(E, n), omega1, omega2, omega3, units)).dot(
        (Del(L[:,n], n) * D1(Del(E, n), omega3, units)))) * Del(L[n,:], n).dot(
    Del(L[:,n], n) * D1(Del(E, n), omega1, units))

    return term21

def gamma_term22(L, I, E, omega1, omega2, omega3, units, n=0):
    """Returns the second term of the second summand in sos expression
    for gamma_eeee

    Input
        xx : np.array
            transition matrix
        E : np.array
            complex eigenenergies (imaginary part is due to damping)
        omega1-3 : np.float
            incident field frequencies
        units : class
            fundamental constants

    Output
        gamma_term : complex
            the second sum in the second set of terms of gamma_eeee
    """

    term22 = units.g**4 * ((Del(L[n,:], n) * D1(Del(E.conjugate(), n), -omega2, units)).dot(
        (Del(L[:,n], n) * D1(Del(E, n), omega1, units)))) * Del(L[n,:], n).dot(
    Del(L[:,n], n) * D1(Del(E, n), omega3, units))

    return term22

def gamma_term23(L, I, E, omega1, omega2, omega3, units, n=0):
    """Returns the third term of the second summand in sos expression
    for gamma_eeee

    Input
        xx : np.array
            transition matrix
        E : np.array
            complex eigenenergies (imaginary part is due to damping)
        omega1-3 : np.float
            incident field frequencies
        units : class
            fundamental constants

    Output
        gamma_term : complex
            the third sum in the second set of terms of gamma_eeee
    """

    term23 = units.g**4 * ((Del(L[n,:], n) * D3(Del(E.conjugate(), n), -omega1,-omega2, -omega3, units)).dot(
        (Del(L[:,n], n) * D1(Del(E.conjugate(), n), -omega3, units)))) * Del(L[n,:], n).dot(
    Del(L[:,n], n) * D1(Del(E.conjugate(), n), -omega1, units))

    return term23

def gamma_term24(L, I, E, omega1, omega2, omega3, units, n=0):
    """Returns the fourth term of the second summand in sos expression
    for gamma_eeee

    Input
        xx : np.array
            transition matrix
        E : np.array
            complex eigenenergies (imaginary part is due to damping)
        omega1-3 : np.float
            incident field frequencies
        units : class
            fundamental constants

    Output
        gamma_term : complex
            the fourth sum in the second set of terms of gamma_eeee
    """

    term24 = units.g**4 * ((Del(L[n,:], n) * D1(Del(E.conjugate(), n), -omega1, units)).dot(
        (Del(L[:,n], n) * D1(Del(E, n), omega2, units)))) * Del(L[n,:], n).dot(
    Del(L[:,n], n) * D1(Del(E.conjugate(), n), -omega3, units))

    return term24

def gamma_term31(L, I, E, omega1, omega2, omega3, units, n=0):
    """Returns the first term of the third summand in sos expression
    for gamma_eeee

    Input
        xx : np.array
            transition matrix
        E : np.array
            complex eigenenergies (imaginary part is due to damping)
        omega1-3 : np.float
            incident field frequencies
        units : class
            fundamental constants

    Output
        gamma_term : complex
            the first sum in the third set of terms of gamma_eeee
    """

    term31 = units.g**4 * (Del(L[n,:], n) * D3(Del(E, n), omega1, omega2, omega3, units)).dot(
        Del(Del(L, n, 0), n, 1).dot((Del(I[:,n], n) * D2(Del(E, n), omega1, omega2, units))
        )) / 2

    return term31

def gamma_term32(L, I, E, omega1, omega2, omega3, units, n=0):
    """Returns the second term of the third summand in sos expression
    for gamma_eeee

    Input
        xx : np.array
            transition matrix
        E : np.array
            complex eigenenergies (imaginary part is due to damping)
        omega1-3 : np.float
            incident field frequencies
        units : class
            fundamental constants

    Output
        gamma_term : complex
            the second sum in the third set of terms of gamma_eeee
    """

    term32 = units.g**4 * (Del(I[n,:], n) * D3(Del(E.conjugate(), n), -omega1, -omega2, -omega3, units)).dot(
        Del(Del(L, n, 0), n, 1).dot((Del(L[:,n], n) * D2(Del(E.conjugate(), n), -omega1, -omega2, units))
        )) / 2

    return term32

def gamma_term33(L, I, E, omega1, omega2, omega3, units, n=0):
    """Returns the first term of the third summand in sos expression
    for gamma_eeee

    Input
        xx : np.array
            transition matrix
        E : np.array
            complex eigenenergies (imaginary part is due to damping)
        omega1-3 : np.float
            incident field frequencies
        units : class
            fundamental constants

    Output
        gamma_term : complex
            the third sum in the third set of terms of gamma_eeee
    """

    term33 = units.g**4 * (Del(L[n,:], n) * D3(Del(E, n), omega1, omega2, omega3, units)).dot(
        Del(Del(I, n, 0), n, 1).dot((Del(L[:,n], n) * D1(Del(E, n), omega1, units))
        )) / 2

    return term33

def gamma_term34(L, I, E, omega1, omega2, omega3, units, n=0):
    """Returns the first term of the third summand in sos expression
    for gamma_eeee

    Input
        xx : np.array
            transition matrix
        E : np.array
            complex eigenenergies (imaginary part is due to damping)
        omega1-3 : np.float
            incident field frequencies
        units : class
            fundamental constants

    Output
        gamma_term : complex
            the fourth sum in the third set of terms of gamma_eeee
    """
    term34 = units.g**4 * (Del(L[n,:], n) * D3(Del(E.conjugate(), n), -omega1, -omega2, -omega3, units)).dot(
        Del(Del(I, n, 0), n, 1).dot((Del(L[:,n], n) * D1(Del(E.conjugate(), n), -omega1, units))
        )) / 2

    return term34

def gamma_term35(L, I, E, omega1, omega2, omega3, units, n=0):
    """Returns the first term of the third summand in sos expression
    for gamma_eeee

    Input
        xx : np.array
            transition matrix
        E : np.array
            complex eigenenergies (imaginary part is due to damping)
        omega1-3 : np.float
            incident field frequencies
        units : class
            fundamental constants

    Output
        gamma_term : complex
            the first sum in the third set of terms of gamma_eeee
    """

    term35 = units.g**4 * (Del(L[n,:], n) * D1(Del(E.conjugate(), n), -omega1, units)).dot(
        Del(Del(L, n, 0), n, 1).dot((Del(I[:,n], n) * D2(Del(E, n), omega2, omega3, units))
        )) / 2

    return term35

def gamma_term36(L, I, E, omega1, omega2, omega3, units, n=0):
    """Returns the second term of the third summand in sos expression
    for gamma_eeee

    Input
        xx : np.array
            transition matrix
        E : np.array
            complex eigenenergies (imaginary part is due to damping)
        omega1-3 : np.float
            incident field frequencies
        units : class
            fundamental constants

    Output
        gamma_term : complex
            the second sum in the third set of terms of gamma_eeee
    """

    term36 = units.g**4 * (Del(I[n,:], n) * D1(Del(E, n), omega1, units)).dot(
        Del(Del(L, n, 0), n, 1).dot((Del(L[:,n], n) * D2(Del(E.conjugate(), n), -omega2, -omega3, units))
        )) / 2

    return term36

def gamma_term41(L, I, E, omega1, omega2, omega3, units, n=0):
    """Returns the second term of the third summand in sos expression
    for gamma_eeee

    Input
        xx : np.array
            transition matrix
        E : np.array
            complex eigenenergies (imaginary part is due to damping)
        omega1-3 : np.float
            incident field frequencies
        units : class
            fundamental constants

    Output
        gamma_term : complex
            the second sum in the third set of terms of gamma_eeee
    """

    term41 = units.g**4 * (Del(I[n,:], n) * D2(Del(E, n), omega2, omega3, units)).dot(
        Del(Del(L, n, 0), n, 1).dot((Del(L[:,n], n) * D1(Del(E, n), omega3, units))
        ))

    return term41

def gamma_term42(L, I, E, omega1, omega2, omega3, units, n=0):
    """Returns the second term of the third summand in sos expression
    for gamma_eeee

    Input
        xx : np.array
            transition matrix
        E : np.array
            complex eigenenergies (imaginary part is due to damping)
        omega1-3 : np.float
            incident field frequencies
        units : class
            fundamental constants

    Output
        gamma_term : complex
            the second sum in the third set of terms of gamma_eeee
    """

    term42 = units.g**4 * (Del(L[n,:], n) * D2(Del(E.conjugate(), n), omega2, omega3, units)).dot(
        Del(Del(L, n, 0), n, 1).dot((Del(I[:,n], n) * D1(Del(E.conjugate(), n), omega3, units))
        ))

    return term42

def gamma_term43(L, I, E, omega1, omega2, omega3, units, n=0):
    """Returns the second term of the third summand in sos expression
    for gamma_eeee

    Input
        xx : np.array
            transition matrix
        E : np.array
            complex eigenenergies (imaginary part is due to damping)
        omega1-3 : np.float
            incident field frequencies
        units : class
            fundamental constants

    Output
        gamma_term : complex
            the second sum in the third set of terms of gamma_eeee
    """

    term43 = units.g**4 * (Del(L[n,:], n) * D1(Del(E.conjugate(), n), omega1, units)).dot(
        Del(Del(I, n, 0), n, 1).dot((Del(L[:,n], n) * D1(Del(E, n), omega3, units))
        ))

    return term43

def gamma_term51(L, I, E, omega1, omega2, omega3, units, n=0):
    """Returns the second term of the third summand in sos expression
    for gamma_eeee

    Input
        xx : np.array
            transition matrix
        E : np.array
            complex eigenenergies (imaginary part is due to damping)
        omega1-3 : np.float
            incident field frequencies
        units : class
            fundamental constants

    Output
        gamma_term : complex
            the second sum in the third set of terms of gamma_eeee
    """
    
    term51 = units.g**4 * Del(I[n,:], n).dot(Del(I[:,n], n) * D2(Del(E, n), omega2, omega3, units)) / 2
    
    return term51

def gamma_term52(L, I, E, omega1, omega2, omega3, units, n=0):
    """Returns the second term of the third summand in sos expression
    for gamma_eeee

    Input
        xx : np.array
            transition matrix
        E : np.array
            complex eigenenergies (imaginary part is due to damping)
        omega1-3 : np.float
            incident field frequencies
        units : class
            fundamental constants

    Output
        gamma_term : complex
            the second sum in the third set of terms of gamma_eeee
    """
    
    term51 = units.g**4 * Del(I[n,:], n).dot(Del(I[:,n], n) * D2(Del(E.conjugate(), n), -omega2, -omega3, units)) / 2
    
    return term51

def permute_gamma_terms(gamma_term, L, I, E, omega, units,n=0):
    """Averages the function `gamma_term` over all permutations of omega1, omega2, and omega3.

    Input
        gamma_term : function
            function that returns one of the terms in sos expression for 
            gamma_eeee
        xx : np.array
            transition matrix
        E : np.array
            eigenenergies Emn
        omega : np.array
            incident field frequencies
        units : class
            fundamental constants
        n : int
            starting state

    Output
        gamma_term : complex
            The average of the term over all frequency permutations
        """

    gamma_term = (1 / 6) * (gamma_term(L, I, E, omega[0], omega[1], omega[2], units,n=0)
        + gamma_term(L, I, E, omega[0], omega[2], omega[1], units,n=0)
        + gamma_term(L, I, E, omega[1], omega[0], omega[2], units,n=0)
        + gamma_term(L, I, E, omega[1], omega[2], omega[0], units,n=0)
        + gamma_term(L, I, E, omega[2], omega[1], omega[0], units,n=0)
        + gamma_term(L, I, E, omega[2], omega[0], omega[1], units,n=0)) 

    return gamma_term