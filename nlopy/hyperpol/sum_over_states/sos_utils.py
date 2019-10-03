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


def get_SOS_operators_summand1and2(gamma_type, X, L, units):
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
    A = np.zeros((3, len(X), len(X))) + 0j
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

def get_SOS_operators_summand4(gamma_type, X, L, I, units):
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
        AI, A1, A2 : np.arrays
            Appropriate operators along the cartesian directions that appear
            in the experssions for gamma. AI is g^2 I / 2, A1 and A2 are either eX or gL
    """
    Types ='em'
    A = np.zeros((3, len(X), len(X))) + 0j
    for i in range(4):
        assert gamma_type[i] in Types, "The character '"+gamma_type[i]+"'' is not a valid type. The only types are electric (e) and magnetic (m)."
    assert gamma_type[0]=='m', "The fourth summand doesn't appear for induced electric dipole."
    assert gamma_type.count('m')>=2, "Fourth summand must have at least one incident magnetic field"
    A[0] = 0.5 * units.g**2 * I
    if gamma_type.count('e')==2:
        A[1] = units.e * X
        A[2] = units.e * X
    elif gamma_type.count('e')==1:
        A[1] = units.g * L
        A[2] = units.e * X
    else:
        A[1] = units.g * L
        A[2] = units.g * L 
    return A

def gamma_term11(O1, O2, O3, O4, E, omega1, omega2, omega3, units, n=0):
    """Returns the first term of the first summand in SOS expression
    for gamma_xxxx

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
    term11 =  (Del(O1[n,:], n) * D3(Del(E,n), omega1, omega2, omega3, units)).dot(
        (Del(Del(O4, n, 0), n, 1) * D2(Del(E, n), omega1, omega2, units)).dot(
            Del(Del(O3, n, 0), n, 1).dot(
                (Del(O2[:,n], n) * D1(Del(E, n), omega1, units)))))
    return term11
   

def gamma_term12(O1, O2, O3, O4, E, omega1, omega2, omega3, units, n=0):
    """Returns the second term of the first summand in SOS expression
    for gamma_xxxx

    Input
        Oi : np.array
            array corresponding to ith index
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
    term12 = (Del(O2[n,:], n) * D1(Del(E.conjugate(),n), -omega1, units)).dot(
        (Del(Del(O1, n, 0), n, 1) * D2(Del(E, n), omega2, omega3, units)).dot(
            Del(Del(O4, n, 0), n, 1).dot(
                (Del(O3[:,n], n) * D1(Del(E, n), omega3, units))
                )
            )
        ) 
    return term12


def gamma_term13(O1, O2, O3, O4, E, omega1, omega2, omega3, units, n=0):
    """Returns the thrid term of the first summand in SOS expression
    for gamma_xxxx

    Input
        Oi : np.array
            array corresponding to ith index
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
    term13 = (Del(O3[n,:], n) * D1(Del(E.conjugate(),n), -omega3, units)).dot(
        (Del(Del(O4, n, 0), n, 1) * D2(Del(E.conjugate(), n), -omega3, -omega2, units)).dot(
            Del(Del(O1, n, 0), n, 1).dot(
                (Del(O2[:,n], n) * D1(Del(E, n), omega1, units))
                )
            )
        )
    return term13

def gamma_term14(O1, O2, O3, O4, E, omega1, omega2, omega3, units, n=0):
    """Returns the fourth term of the first summand in SOS expression
    for gamma_xxxx

    Input
        Oi : np.array
            array corresponding to ith index
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
    term14 = (Del(O3[n,:], n) * D1(Del(E.conjugate(), n), -omega2, units)).dot(
        (Del(Del(O2, n, 0), n, 1) * D2(Del(E.conjugate(), n), -omega1, -omega2, units)).dot(
            Del(Del(O4, n, 0), n, 1).dot(
                (Del(O1[:,n], n) * D3(Del(E.conjugate(), n), -omega1, -omega2, -omega3, units))
                )
            )
        ) 
    return term14

def gamma_term21(O1, O2, O3, O4, E, omega1, omega2, omega3, units, n=0):
    """Returns the first term of the second summand in sos expression
    for gamma_xxxx

    Input
        Oi : np.array
            array corresponding to ith index
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
    term21 = ((Del(O1[n,:], n) * D3(Del(E, n), omega1, omega2, omega3, units)).dot(
        (Del(O4[:,n], n) * D1(Del(E, n), omega3, units)))) * Del(O3[n,:], n).dot(
    Del(O2[:,n], n) * D1(Del(E, n), omega1, units))
    return term21

def gamma_term22(O1, O2, O3, O4, E, omega1, omega2, omega3, units, n=0):
    """Returns the second term of the second summand in sos expression
    for gamma_xxxx

    Input
        Oi : np.array
            array corresponding to ith index
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
    term22 = ((Del(O3[n,:], n) * D1(Del(E.conjugate(), n), -omega2, units)).dot(
        (Del(O2[:,n], n) * D1(Del(E, n), omega1, units)))) * Del(O1[n,:], n).dot(
    Del(O4[:,n], n) * D1(Del(E, n), omega3, units))
    return term22

def gamma_term23(O1, O2, O3, O4, E, omega1, omega2, omega3, units, n=0):
    """Returns the third term of the second summand in sos expression
    for gamma_xxxx

    Input
        Oi : np.array
            array corresponding to ith index
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
    term23 = ((Del(O4[n,:], n) * D3(Del(E.conjugate(), n), -omega1,-omega2, -omega3, units)).dot(
        (Del(O1[:,n], n) * D1(Del(E.conjugate(), n), -omega3, units)))) * Del(O2[n,:], n).dot(
    Del(O3[:,n], n) * D1(Del(E.conjugate(), n), -omega1, units))
    return term23

def gamma_term24(O1, O2, O3, O4, E, omega1, omega2, omega3, units, n=0):
    """Returns the fourth term of the second summand in sos expression
    for gamma_xxxx

    Input
        Oi : np.array
            array corresponding to ith index
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
    term24 = ((Del(O2[n,:], n) * D1(Del(E.conjugate(), n), -omega1, units)).dot(
        (Del(O3[:,n], n) * D1(Del(E, n), omega2, units)))) * Del(O4[n,:], n).dot(
    Del(O1[:,n], n) * D1(Del(E.conjugate(), n), -omega3, units))
    return term24

def permute_gamma_4op_terms(gamma_term, O1, O2, O3, O4, E, omega, units, gamma_type, n=0):
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
        gamma_type : string
            string specifying which gamma we are computiner ('eeee', 'meem', 'mmme', etc.)
        n : int
            starting state

    Output
        gamma_term : complex
            The average of the term over all frequency permutations
        """

    Gamma_term = (1 / 6) * (gamma_term(O1, O2, O3, O4, E, omega[0], omega[1], omega[2], units, n)
        + gamma_term(O1, O3, O4, O2, E, omega[1], omega[2], omega[0], units, n)
        + gamma_term(O1, O4, O2, O3, E, omega[2], omega[0], omega[1], units, n)
        + gamma_term(O1, O3, O2, O4, E, omega[1], omega[0], omega[2], units, n)
        + gamma_term(O1, O2, O4, O3, E, omega[0], omega[2], omega[1], units, n)
        + gamma_term(O1, O4, O3, O2, E, omega[2], omega[1], omega[0], units, n)
        ) 
    return Gamma_term

def permute_gamma_terms_123(gamma_term, E, X, L, I, omega, units, gamma_type, n=0):
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
        gamma_type : string
            string specifying which gamma we are computiner ('eeee', 'meem', 'mmme', etc.)
        n : int
            starting state

    Output
        gamma_term : complex
            The average of the term over all frequency permutations
        """

    Gamma_term = (1 / 6) * (gamma_term(X, L, E, omega[0], omega[1], omega[2], units, gamma_type, n)
        + gamma_term(X, L, E, omega[0], omega[2], omega[1], units, gamma_type, n)
        + gamma_term(X, L, E, omega[1], omega[0], omega[2], units, gamma_type, n)
        + gamma_term(X, L, E, omega[1], omega[2], omega[0], units, gamma_type, n)
        + gamma_term(X, L, E, omega[2], omega[1], omega[0], units, gamma_type, n)
        + gamma_term(X, L, E, omega[2], omega[0], omega[1], units, gamma_type, n)) 
    return Gamma_term

def gamma_term31(O1, O2, O3, E, omega1, omega2, omega3, units, gamma_type, n=0):
    """Returns the first term of the third summand in sos expression
    for gamma_xxxx

    Input
        O1-3 : np.array
            operators
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
    term31 = (Del(O1[n,:], n) * D3(Del(E, n), omega1, omega2, omega3, units)).dot(
        Del(Del(O2, n, 0), n, 1).dot((Del(O3[:,n], n) * D2(Del(E, n), omega1, omega2, units))
        ))
    return term31

def gamma_term32(O1, O2, O3, E, omega1, omega2, omega3, units, gamma_type, n=0):
    """Returns the second term of the third summand in sos expression
    for gamma_xxxx

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
    term32 = (Del(O1[n,:], n) * D3(Del(E.conjugate(), n), -omega1, -omega2, -omega3, units)).dot(
        Del(Del(O2, n, 0), n, 1).dot((Del(O3[:,n], n) * D2(Del(E.conjugate(), n), -omega2, -omega3, units))
        ))

    return term32

def gamma_term33(O1, O2, O3, E, omega1, omega2, omega3, units, gamma_type, n=0):
    """Returns the third term of the third summand in sos expression
    for gamma_xxxx

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
    term33 = (Del(O1[n,:], n) * D3(Del(E, n), omega1, omega2, omega3, units)).dot(
        Del(Del(O2, n, 0), n, 1).dot((Del(O3[:,n], n) * D1(Del(E, n), omega1, units))
        ))
    return term33

def gamma_term34(O1, O2, O3, E, omega1, omega2, omega3, units, gamma_type, n=0):
    """Returns the fourth term of the third summand in sos expression
    for gamma_xxxx

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
    term34 = (Del(Ax[n,:], n) * D3(Del(E.conjugate(), n), -omega1, -omega2, -omega3, units)).dot(
        Del(Del(AI, n, 0), n, 1).dot((Del(Ai[:,n], n) * D1(Del(E.conjugate(), n), -omega1, units))
        ))
    return term34

def gamma_term35(O1, O2, O3, E, omega1, omega2, omega3, units, gamma_type, n=0):
    """Returns the fifth term of the third summand in sos expression
    for gamma_xxxx

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
    term35 = (Del(Ax[n,:], n) * D1(Del(E.conjugate(), n), -omega1, units)).dot(
        Del(Del(Ai, n, 0), n, 1).dot((Del(AI[:,n], n) * D2(Del(E, n), omega2, omega3, units))
        ))

    return term35

def gamma_term36(O1, O2, O3, E, omega1, omega2, omega3, units, gamma_type, n=0):
    """Returns the sixth term of the third summand in sos expression
    for gamma_xxxx

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
    term36 = (Del(AI[n,:], n) * D1(Del(E, n), omega1, units)).dot(
        Del(Del(Ai, n, 0), n, 1).dot((Del(Ax[:,n], n) * D2(Del(E.conjugate(), n), -omega2, -omega3, units))
        ))

    return term36

def permute_gamma_summand3_terms(gamma_term, O1, O2, O3, E, omega, units, gamma_type, n=0):
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
        gamma_type : string
            string specifying which gamma we are computiner ('eeee', 'meem', 'mmme', etc.)
        n : int
            starting state

    Output
        gamma_term : complex
            The average of the term over all frequency permutations
        """
    if gamma_type=='mmmm':
        Gamma_term = (1 / 6) * (gamma_term(O1, O2, O3, E, omega[0], omega[1], omega[2], units, n)
            + gamma_term(O1, O2, O3, E, omega[1], omega[2], omega[0], units, n)
            + gamma_term(O1, O2, O3, E, omega[2], omega[0], omega[1], units, n)
            + gamma_term(O1, O2, O3, E, omega[1], omega[0], omega[2], units, n)
            + gamma_term(O1, O2, O3, E, omega[0], omega[2], omega[1], units, n)
            + gamma_term(O1, O2, O3, E, omega[2], omega[1], omega[0], units, n)
            ) 
        return Gamma_term
    if gamma_type=='mmme':
        Gamma_term = (1 / 6) * (gamma_term(O1, O2, O3, E, omega[2], omega[1], omega[0], units, n)
            + gamma_term(O1, O2, O3, E, omega[2], omega[0], omega[1], units, n)
            ) 
        return Gamma_term
    if gamma_type=='mmem':
        Gamma_term = (1 / 6) * (gamma_term(O1, O2, O3, E, omega[1], omega[2], omega[0], units, n)
            + gamma_term(O1, O2, O3, E, omega[1], omega[0], omega[2], units, n)
            ) 
        return Gamma_term
    if gamma_type=='memm':
        Gamma_term = (1 / 6) * (gamma_term(O1, O2, O3, E, omega[0], omega[1], omega[2], units, n)
            + gamma_term(O1, O2, O3, E, omega[0], omega[2], omega[1], units, n)
            ) 
        return Gamma_term
    else:
        assert False, gamma_type+"Not implemented yet."

def gamma_term41(O1, O2, O3, E, omega1, omega2, omega3, units, n=0):
    """Returns the first term of the fourth summand in sos expression
    for gamma_mxxxx. 

    Input
        O : np.array
            operators
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

    term41 = units.g**4 * (Del(O1[n,:], n) * D2(Del(E, n), omega2, omega3, units)).dot(
        Del(Del(O2, n, 0), n, 1).dot((Del(O3[:,n], n) * D1(Del(E, n), omega3, units))
        ))

    return term41

def gamma_term42(O1, O2, O3, E, omega1, omega2, omega3, units, n=0):
    """Returns the second term of the third summand in sos expression
    for gamma_eeee

    Input
        O : np.array
            operators
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

    term42 = units.g**4 * (Del(O1[n,:], n) * D2(Del(E.conjugate(), n), -omega2, -omega3, units)).dot(
        Del(Del(O2, n, 0), n, 1).dot((Del(O3[:,n], n) * D1(Del(E.conjugate(), n), -omega3, units))
        ))

    return term42

def gamma_term43(O1, O2, O3, E, omega1, omega2, omega3, units, n=0):
    """Returns the second term of the third summand in sos expression
    for gamma_eeee

    Input
        O : np.array
            operators
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

    term43 = units.g**4 * (Del(O1[n,:], n) * D1(Del(E.conjugate(), n), -omega2, units)).dot(
        Del(Del(O2, n, 0), n, 1).dot((Del(O3[:,n], n) * D1(Del(E, n), omega3, units))
        ))

    return term43

    def permute_gamma_summand4_terms(gamma_term, O1, O2, O3, E, omega, units, gamma_type, n=0):
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
        gamma_type : string
            string specifying which gamma we are computiner ('eeee', 'meem', 'mmme', etc.)
        n : int
            starting state

    Output
        gamma_term : complex
            The average of the term over all frequency permutations
        """
    if gamma_type=='mmmm':
        Gamma_term = (1 / 6) * (gamma_term(O1, O2, O3, E, omega[0], omega[1], omega[2], units, n)
            + gamma_term(O1, O2, O3, E, omega[1], omega[2], omega[0], units, n)
            + gamma_term(O1, O2, O3, E, omega[2], omega[0], omega[1], units, n)
            + gamma_term(O1, O2, O3, E, omega[1], omega[0], omega[2], units, n)
            + gamma_term(O1, O2, O3, E, omega[0], omega[2], omega[1], units, n)
            + gamma_term(O1, O2, O3, E, omega[2], omega[1], omega[0], units, n)
            ) 
        return Gamma_term
    elif gamma_type=='mmme':
        Gamma_term = (1 / 6) * (gamma_term(O1, O2, O3, E, omega[0], omega[1], omega[2], units, n)
            + gamma_term(O1, O2, O3, E, omega[0], omega[2], omega[1], units, n)
            + gamma_term(O1, O2, O3, E, omega[1], omega[0], omega[2], units, n)
            + gamma_term(O1, O2, O3, E, omega[1], omega[2], omega[0], units, n)
            ) 
        return Gamma_term
    elif gamma_type=='mmem':
        Gamma_term = (1 / 6) * (gamma_term(O1, O2, O3, E, omega[0], omega[1], omega[2], units, n)
            + gamma_term(O1, O2, O3, E, omega[], omega[0], omega[1], units, n)
            + gamma_term(O1, O2, O3, E, omega[2], omega[0], omega[1], units, n)
            + gamma_term(O1, O2, O3, E, omega[2], omega[1], omega[0], units, n)
            ) 
        return Gamma_term
    elif gamma_type=='memm':
        Gamma_term = (1 / 6) * (gamma_term(O1, O2, O3, E, omega[1], omega[2], omega[0], units, n)
            + gamma_term(O1, O2, O3, E, omega[1], omega[0], omega[2], units, n)
            + gamma_term(O1, O2, O3, E, omega[2], omega[0], omega[1], units, n)
            + gamma_term(O1, O2, O3, E, omega[2], omega[1], omega[0], units, n)
            ) 
        return Gamma_term
    else:
        assert False, gamma_type+"Not implemented yet."

def gamma_term51(O1, O2, E, omega1, omega2, omega3, units, n=0):
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
    
    term51 = units.g**4 * Del(O1[n,:], n).dot(Del(O2[:,n], n) * D2(Del(E, n), omega2, omega3, units)) / 2
    
    return term51

def gamma_term52(O1, O2, E, omega1, omega2, omega3, units, n=0):
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
    
    term51 = units.g**4 * Del(O1[n,:], n).dot(Del(O2[:,n], n) * D2(Del(E.conjugate(), n), -omega2, -omega3, units)) / 2
    
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
