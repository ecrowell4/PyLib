import numpy as np
import scipy.integrate as sp_integrate
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


def get_SOS_operators_summand12(gamma_type, X, L, units):
    """Given the type of (hyper) polarizability to be computed,
    returns the operators corresponding to dH/dFi for the first two summands
    in gamma_xxxx.

    For example, if type='eeem', then it is assumed we are computing the electric
    dipole moment along i due to electric fields along j,k and a magnetic
    field along r. 

    If type=='memm', then we are computing the magnetic dipole moment along i due to 
    and electric field along j and magnetic fields along k,r.

    Note that the coupling constants are included as well, so there will be no
    factors of e^n g^m in from of the gamma_terms.

    Input
        gamma_type : string
            specifieds which (hyper) polarizability to compute.
            E.g. 'eeee' would imply the all electric gamma
        X : np.array
            transition matrix
        L : np.array
            angular momentum matrix
        units : class
            fundamental constants

    Output
        Ai, Aj, Ak, Ar : np.arrays
            Appropriate operators along the cartesian directions that appear
            in the experssions for gamma.
    """
    Types ='em'
    A = np.zeros((4, len(X), len(X))) + 0j
    for i in range(4):
        assert gamma_type[i] in Types, "The character '"+gamma_type[i]+"'' is not a valid type. The only types are electric (e) and magnetic (m)."
        if gamma_type[i]=='e':
        	A[i] = units.e * X
        elif gamma_type[i]=='m':
        	A[i] = units.g * L
    return A

def get_SOS_operators_summand3(gamma_type, X, L, I, units):
    """Given the type of (hyper) polarizability to be computed,
    returns the operators corresponding to dH/dFi for the first two summands
    in gamma_xxxx.

    For example, if type='eeem', then it is assumed we are computing the electric
    dipole moment along i due to electric fields along j,k and a magnetic
    field along r. 

    If type=='memm', then we are computing the magnetic dipole moment along i due to 
    and electric field along j and magnetic fields along k,r.

    Note that the coupling constants are included as well, so there will be no
    factors of e^n g^m in from of the gamma_terms.

    Input
        gamma_type : string
            specifieds which (hyper) polarizability to compute.
            E.g. 'eeee' would imply the all electric gamma
        X : np.array
            transition matrix
        L : np.array
            angular momentum matrix
        I : np.array
            matrix elements of Izz
        units : class
            fundamental constants

    Output
        Ai, Ax, AI : np.arrays
            Appropriate operators along the cartesian directions that appear
            in the experssions for gamma. Ai, Ax are either eX or gL and AI = g^2 I / 2
    """
    Types ='em'
    A = np.zeros((0, len(X), len(X))) + 0j
    for i in range(4):
        assert gamma_type[i] in Types, "The character '"+gamma_type[i]+"'' is not a valid type. The only types are electric (e) and magnetic (m)."
    if gamma_type[0]=='e':
    	A[0] = units.e * X
    elif gamma_type[0]=='m':
    	A[0] = units.g * L
    if gamma_type[1:].count('e')==0:
    	A[1] = units.g * L
    elif gamma_type[1:].count('e')==1:
    	A[1] = units.e * X
    A[2] = 0.5 * units.g**2 * I
    return A

   
def gamma_term11(X, L, E, omega1, omega2, omega3, units, gamma_type, n=0):
    """Returns the first term of the first summand in SOS expression
    for gamma, as written [FILL IN LOCATION]

    Input
        X : np.array
            transition matrix
        L : np.array
            angular momentum matrix
        E : np.array
            complex eigenenergies (imaginary part is due to damping)
        omega1-3 : np.float
            incident field frequencies
        units : class
            fundamental constants
        gamma_type : string
            which gamma to compute, of the form 'xxxx', where x = m,e.
        n : int
            which state system is in. Default is ground state (n=0)

    Output
        gamma_term : complex
            the first sum in the first set of terms of gamma
    """
    assert gamma_type != None, "Must specify which gamma you want: 'eeee', 'eeem', etc."
    Ai, Aj, Ak, Ar = get_SOS_operators_summand12(gamma_type, X, L, units)
    term11 =  (Del(Ai[n,:], n) * D3(Del(E,n), omega1, omega2, omega3, units)).dot(
        (Del(Del(Ar, n, 0), n, 1) * D2(Del(E, n), omega1, omega2, units)).dot(
            Del(Del(Ak, n, 0), n, 1).dot(
                (Del(Aj[:,n], n) * D1(Del(E, n), omega1, units)))))
    return term11

def gamma_term12(X, L, E, omega1, omega2, omega3, units, gamma_type, n=0):
    """Returns the second term of the first summand in SOS expression
    for gamma_eee, as written [FILL IN LOCATION]

    Input
        X : np.array
            transition matrix
        L : np.array
            angular momentum matrix
        E : np.array
            complex eigenenergies (imaginary part is due to damping)
        omega1-3 : np.float
            incident field frequencies
        units : class
            fundamental constants
        gamma_type : string
            which gamma to compute, of the form 'xxxx', where x = m,e.
        n : int
            which state system is in. Default is ground state (n=0)

    Output
        gamma_term : complex
            the second sum in the first set of terms of gamma_eeee
    """
    assert gamma_type != None, "Must specify which gamma you want: 'eeee', 'eeem', etc."
    Ai, Aj, Ak, Ar = get_SOS_operators_summand12(gamma_type, X, L, units)
    term12 = (Del(Aj[n,:], n) * D1(Del(E.conjugate(),n), -omega1, units)).dot(
        (Del(Del(Ai, n, 0), n, 1) * D2(Del(E, n), omega2, omega3, units)).dot(
            Del(Del(Ar, n, 0), n, 1).dot(
                (Del(Ak[:,n], n) * D1(Del(E, n), omega3, units))
                )
            )
        ) 
    return term12

def gamma_term13(X, L, E, omega1, omega2, omega3, units, gamma_type, n=0):
    """Returns the thrid term of the first summand in SOS expression
    for gamma_eee, as written [FILL IN LOCATION]

    Input
         X : np.array
            transition matrix
        L : np.array
            angular momentum matrix
        E : np.array
            complex eigenenergies (imaginary part is due to damping)
        omega1-3 : np.float
            incident field frequencies
        units : class
            fundamental constants
        gamma_type : string
            which gamma to compute, of the form 'xxxx', where x = m,e.
        n : int
            which state system is in. Default is ground state (n=0)

    Output
        gamma_term : complex
            the third sum in the first set of terms of gamma_eeee
    """
    assert gamma_type != None, "Must specify which gamma you want: 'eeee', 'eeem', etc."
    Ai, Aj, Ak, Ar = get_SOS_operators_summand12(gamma_type, X, L, units)
    term13 = (Del(Ak[n,:], n) * D1(Del(E.conjugate(),n), -omega3, units)).dot(
        (Del(Del(Ar, n, 0), n, 1) * D2(Del(E.conjugate(), n), -omega3, -omega2, units)).dot(
            Del(Del(Ai, n, 0), n, 1).dot(
                (Del(Aj[:,n], n) * D1(Del(E, n), omega1, units))
                )
            )
        )
    return term13

def gamma_term14(X, L, E, omega1, omega2, omega3, units, gamma_type, n=0):
    """Returns the fourth term of the first summand in SOS expression
    for gamma_eee, as written [FILL IN LOCATION]

    Input
        X : np.array
            transition matrix
        L : np.array
            angular momentum matrix
        E : np.array
            complex eigenenergies (imaginary part is due to damping)
        omega1-3 : np.float
            incident field frequencies
        units : class
            fundamental constants
        gamma_type : string
            which gamma to compute, of the form 'xxxx', where x = m,e.
        n : int
            which state system is in. Default is ground state (n=0)

    Output
        gamma_term : complex
            the fourth sum in the first set of terms of gamma_eeee
    """
    assert gamma_type != None, "Must specify which gamma you want: 'eeee', 'eeem', etc."
    Ai, Aj, Ak, Ar = get_SOS_operators_summand12(gamma_type, X, L, units)
    term14 = (Del(Ak[n,:], n) * D1(Del(E.conjugate(), n), -omega2, units)).dot(
        (Del(Del(Aj, n, 0), n, 1) * D2(Del(E.conjugate(), n), -omega1, -omega2, units)).dot(
            Del(Del(Ar, n, 0), n, 1).dot(
                (Del(Ai[:,n], n) * D3(Del(E.conjugate(), n), -omega1, -omega2, -omega3, units))
                )
            )
        ) 
    return term14

def gamma_term21(X, L, E, omega1, omega2, omega3, units, gamma_type, n=0):
    """Returns the first term of the second summand in sos expression
    for gamma_eeee

    Input
        X : np.array
            transition matrix
        L : np.array
            angular momentum matrix
        E : np.array
            complex eigenenergies (imaginary part is due to damping)
        omega1-3 : np.float
            incident field frequencies
        units : class
            fundamental constants
        gamma_type : string
            which gamma to compute, of the form 'xxxx', where x = m,e.
        n : int
            which state system is in. Default is ground state (n=0)

    Output
        gamma_term : complex
            the first sum in the second set of terms of gamma_eeee
    """
    assert gamma_type != None, "Must specify which gamma you want: 'eeee', 'eeem', etc."
    Ai, Aj, Ak, Ar = get_SOS_operators_summand12(gamma_type, X, L, units)
    term21 = ((Del(Ai[n,:], n) * D3(Del(E, n), omega1, omega2, omega3, units)).dot(
        (Del(Ar[:,n], n) * D1(Del(E, n), omega3, units)))) * Del(Ak[n,:], n).dot(
    Del(Aj[:,n], n) * D1(Del(E, n), omega1, units))
    return term21

def gamma_term22(X, L, E, omega1, omega2, omega3, units, gamma_type, n=0):
    """Returns the second term of the second summand in sos expression
    for gamma_eeee

    Input
        X : np.array
            transition matrix
        L : np.array
            angular momentum matrix
        E : np.array
            complex eigenenergies (imaginary part is due to damping)
        omega1-3 : np.float
            incident field frequencies
        units : class
            fundamental constants
        gamma_type : string
            which gamma to compute, of the form 'xxxx', where x = m,e.
        n : int
            which state system is in. Default is ground state (n=0)

    Output
        gamma_term : complex
            the second sum in the second set of terms of gamma_eeee
    """
    assert gamma_type != None, "Must specify which gamma you want: 'eeee', 'eeem', etc."
    Ai, Aj, Ak, Ar = get_SOS_operators_summand12(gamma_type, X, L, units)
    term22 = ((Del(Ai[n,:], n) * D1(Del(E.conjugate(), n), -omega2, units)).dot(
        (Del(Ar[:,n], n) * D1(Del(E, n), omega1, units)))) * Del(Ak[n,:], n).dot(
    Del(Aj[:,n], n) * D1(Del(E, n), omega3, units))
    return term22

def gamma_term23(X, L, E, omega1, omega2, omega3, units, gamma_type, n=0):
    """Returns the third term of the second summand in sos expression
    for gamma_eeee

    Input
        X : np.array
            transition matrix
        L : np.array
            angular momentum matrix
        E : np.array
            complex eigenenergies (imaginary part is due to damping)
        omega1-3 : np.float
            incident field frequencies
        units : class
            fundamental constants
        gamma_type : string
            which gamma to compute, of the form 'xxxx', where x = m,e.
        n : int
            which state system is in. Default is ground state (n=0)

    Output
        gamma_term : complex
            the third sum in the second set of terms of gamma_eeee
    """
    assert gamma_type != None, "Must specify which gamma you want: 'eeee', 'eeem', etc."
    Ai, Aj, Ak, Ar = get_SOS_operators_summand12(gamma_type, X, L, units)
    term23 = ((Del(Aj[n,:], n) * D3(Del(E.conjugate(), n), -omega1,-omega2, -omega3, units)).dot(
        (Del(Ak[:,n], n) * D1(Del(E.conjugate(), n), -omega3, units)))) * Del(Ar[n,:], n).dot(
    Del(Ai[:,n], n) * D1(Del(E.conjugate(), n), -omega1, units))
    return term23

def gamma_term24(X, L, E, omega1, omega2, omega3, units, gamma_type, n=0):
    """Returns the fourth term of the second summand in sos expression
    for gamma_eeee

    Input
        X : np.array
            transition matrix
        L : np.array
            angular momentum matrix
        E : np.array
            complex eigenenergies (imaginary part is due to damping)
        omega1-3 : np.float
            incident field frequencies
        units : class
            fundamental constants
        gamma_type : string
            which gamma to compute, of the form 'xxxx', where x = m,e.
        n : int
            which state system is in. Default is ground state (n=0)

    Output
        gamma_term : complex
            the fourth sum in the second set of terms of gamma_eeee
    """
    assert gamma_type != None, "Must specify which gamma you want: 'eeee', 'eeem', etc."
    Ai, Aj, Ak, Ar = get_SOS_operators_summand12(gamma_type, X, L, units)
    term24 = ((Del(Ar[n,:], n) * D1(Del(E.conjugate(), n), -omega1, units)).dot(
        (Del(Ai[:,n], n) * D1(Del(E, n), omega2, units)))) * Del(Aj[n,:], n).dot(
    Del(Ak[:,n], n) * D1(Del(E.conjugate(), n), -omega3, units))
    return term24

def gamma_term31(X, L, I, E, omega1, omega2, omega3, units, gamma_type, n=0):
    """Returns the first term of the third summand in sos expression
    for gamma_eeee

    Input
        X : np.array
            transition matrix
        L : np.array
            angular momentum matrix
        I : np.array
            matrix elements of Izz, 
        E : np.array
            complex eigenenergies (imaginary part is due to damping)
        omega1-3 : np.float
            incident field frequencies
        units : class
            fundamental constants
        gamma_type : string
            which gamma to compute, of the form 'xxxx', where x = m,e.
        n : int
            which state system is in. Default is ground state (n=0)

    Output
        gamma_term : complex
            the first sum in the third set of terms of gamma_eeee
    """
    Ai, Ax, AI = get_SOS_operators_summand3(gamma_type, X, L, I, units) 
    term31 = (Del(Ai[n,:], n) * D3(Del(E, n), omega1, omega2, omega3, units)).dot(
        Del(Del(Ax, n, 0), n, 1).dot((Del(AI[:,n], n) * D2(Del(E, n), omega1, omega2, units))
        ))
    return term31

def gamma_term32(X, L, I, E, omega1, omega2, omega3, units, gamma_type, n=0):
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
    Ai, Ax, AI = get_SOS_operators_summand3(gamma_type, X, L, I, units)
    term32 = (Del(AI[n,:], n) * D3(Del(E.conjugate(), n), -omega1, -omega2, -omega3, units)).dot(
        Del(Del(Ax, n, 0), n, 1).dot((Del(Ai[:,n], n) * D2(Del(E.conjugate(), n), -omega1, -omega2, units))
        ))

    return term32

def gamma_term33(X, L, I, E, omega1, omega2, omega3, units, gamma_type, n=0):
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
    Ai, Ax, AI = get_SOS_operators_summand3(gamma_type, X, L, I, units)
    term33 = (Del(Ai[n,:], n) * D3(Del(E, n), omega1, omega2, omega3, units)).dot(
        Del(Del(AI, n, 0), n, 1).dot((Del(Ax[:,n], n) * D1(Del(E, n), omega1, units))
        ))
    return term33

def gamma_term34(X, L, I, E, omega1, omega2, omega3, units, gamma_type, n=0):
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
    Ai, Ax, AI = get_SOS_operators_summand3(gamma_type, X, L, I, units)
    term34 = (Del(Ax[n,:], n) * D3(Del(E.conjugate(), n), -omega1, -omega2, -omega3, units)).dot(
        Del(Del(AI, n, 0), n, 1).dot((Del(Ai[:,n], n) * D1(Del(E.conjugate(), n), -omega1, units))
        ))
    return term34

def gamma_term35(X, L, I, E, omega1, omega2, omega3, units, gamma_type, n=0):
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
    Ai, Ax, AI = get_SOS_operators_summand3(gamma_type, X, L, I, units)
    term35 = (Del(Ax[n,:], n) * D1(Del(E.conjugate(), n), -omega1, units)).dot(
        Del(Del(Ai, n, 0), n, 1).dot((Del(AI[:,n], n) * D2(Del(E, n), omega2, omega3, units))
        ))

    return term35

def gamma_term36(X, L, I, E, omega1, omega2, omega3, units, gamma_type, n=0):
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
    Ai, Ax, AI = get_SOS_operators_summand3(gamma_type, X, L, I, units)
    term36 = (Del(AI[n,:], n) * D1(Del(E, n), omega1, units)).dot(
        Del(Del(Ai, n, 0), n, 1).dot((Del(Ax[:,n], n) * D2(Del(E.conjugate(), n), -omega2, -omega3, units))
        ))

    return term36

def gamma_term41_m(L, I, E, omega1, omega2, omega3, units, n=0):
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

def gamma_term42_m(L, I, E, omega1, omega2, omega3, units, n=0):
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

def gamma_term43_m(L, I, E, omega1, omega2, omega3, units, n=0):
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

def gamma_term51_m(L, I, E, omega1, omega2, omega3, units, n=0):
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

def gamma_term52_m(L, I, E, omega1, omega2, omega3, units, n=0):
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

def permute_gamma_terms_m(gamma_term, L, I, E, omega, units,n=0):
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

    Gamma_term = (1 / 6) * (gamma_term(L, I, E, omega[0], omega[1], omega[2], units,n=0)
        + gamma_term(L, I, E, omega[0], omega[2], omega[1], units,n=0)
        + gamma_term(L, I, E, omega[1], omega[0], omega[2], units,n=0)
        + gamma_term(L, I, E, omega[1], omega[2], omega[0], units,n=0)
        + gamma_term(L, I, E, omega[2], omega[1], omega[0], units,n=0)
        + gamma_term(L, I, E, omega[2], omega[0], omega[1], units,n=0)) 

    return Gamma_term

def get_F_DL(x, psi0, units):
    """Returns the operator F_0(x) as defined in DalGarno Lewis
    perturbation theory. See reference [1]

    Input
        x : np.array
            position grid
        psi0 : np.array
            ground state wavefunction
        units : Class
            fundamental constants

    Output
        F : np.array
            F operator from DL PT

    [1] S. Mossman, R. Lytel, M. G. Kuzk Dalgarno-Lewis perturbation theory
    for nonlinear optics, Vol. 33, No. 12, (2016)
    """

    dx = x[1] - x[0]

    # Compute mean position of unperturbed ground state x00 = <0|x|0>
    x00 = sp_integrate.simps(psi0.conjugate() * x * psi0, dx=dx)

    # The first integral has a definite lower bound, but a variable upper
    # bound. Thus, we represent it as an array of integrals, each
    # beginning at the left of the domain.
    int1 = np.array([np.trapz( (x[:i]-x00)*psi0[:i]**2, x[:i] ) 
                            for i in np.arange(1, len(x)+1)])

    # Now we integrate int1 to get F(x)
    F = 2*units.m/units.hbar**2 * np.array(
        [np.trapz( psi0[1:i]**(-2)*int1[1:i], x[1:i])
                                for i in np.arange(2, len(x))]
                              )
    F = np.append(0,np.append(F,0))
    
    F = F - sp_integrate.simps(F*psi0**2,x)
    return F
