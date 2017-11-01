import numpy as np

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
    
    # Take E -> E - E0
    E = E - E[0]
    
    # Compute and return the propogator
    return 1 / (E[1:] - units.hbar*omega)

def D2(E, omega1, omega2, units):
    """Returns propogator for a second order process.
    
    Input
    	E : np.array
        	array of eigenergies.
        	Note: if omega is take as negative, then user 
        	should pass E.conjugate() instead of E.
    omega1, omega2 : float
        input frequencies
    units : class
        class whose attributes are the fundamental constants hbar, e, m, c, etc

    Output
    	D2 : np.array
    		propogator for second order process
    """
    E = E - E[0]
    return 1 / (E[1:] - units.hbar*omega1 - units.hbar*omega2)


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

