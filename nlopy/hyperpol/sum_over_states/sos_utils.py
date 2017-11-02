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

def project(x, f, g):
    """Evaluates the projection of function f onto function g, c_n = <f | g>.
    
    Input
        x : np.array
            Spatial grid
        f : np.array
            Function to be projected
        g : np.array
            Function on which to project
        
    Output
        c_n : complex
            Component of psi along phi_n
    """
    
    c_n = np.trapz(f.conjugate() * g, x)
    
    return c_n

def lift_degen(V_prime, x, psi0, psi1):
    """Returns a linear combinations of two degnerate states that diagonalizes
    the modified perturbation. V
    
    Input
        V_prime : np.array
            Matrix representation of modified perturbation in unperturbed 
            degenerate subspace.
        x : np.array
            Spatial grid
        psi0, psi1 : np.array
            Quasi degenerate unperturbed eigenfunctions
        
    Output
        psi_prime : np.array
            Linear combinations of psi0, psi1 that diagonlizes the perturbation.
        E1 : np.array(2) 
            the first order energy corrections to the two states psi1, psi2.
    """

    # The eigenfunctions of modified perturbation will diagonalize it:
    E1, coeffs = np.linalg.eig(V_prime)
    
    coeffs = coeffs.transpose()
    
    psi_prime = np.zeros((2, len(x)))
    for i in range(2):
        # Create linear combinations
        psi_prime[i] = coeffs[i,0]*psi0 + coeffs[i,1]*psi1
        # Normalize the resulting wavefunctions
        psi_prime[i] /= np.sqrt(np.trapz(psi_prime[i].conjugate() * psi_prime[i], x))
    
    return psi_prime, E1
    
    
    