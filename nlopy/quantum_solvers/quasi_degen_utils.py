
import numpy as np
from nlopy.quantum_solvers import solver_utils
import copy

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


def lin_comb(H1_tilde_unprimed, x, psi):
    """Returns a linear combinations of two degnerate states that diagonalizes
    the modified perturbation. V
    
    Input
        H1_tilde_unprimed : np.array
            Matrix representation of modified perturbation in unperturbed 
            degenerate subspace.
        x : np.array
            Spatial grid
        psi : np.array
            Original unperturbed eigenfunctions with quasi degeneracy between
            ground and first excited state.
        
    Output
        psi_prime : np.array
            Linear combinations of psi0, psi1 that diagonlizes the perturbation.
        E1 : np.array(2) 
            the first order energy corrections to the two states psi1, psi2.
    """

    # The eigenfunctions of modified perturbation will diagonalize it:
    E1, coeffs = np.linalg.eig(H1_tilde_unprimed)
    
    coeffs = coeffs.transpose()
    
    # Most of the states will be unaffected, so we initially copy the entire array
    psi_prime = copy.copy(psi)
    
    # The ground and first excited states must be modified in order to diagonalize
    #   the modified perturbation:
    for i in range(2):
        # Normalize coefficients. This prevents us from having to normalize the
        #   wavefunctions.
        coeffs[i] /= np.sqrt(abs(coeffs[i,0])**2 + abs(coeffs[i,1])**2)
        # Create linear combinations
        psi_prime[i] = (coeffs[i,0]*psi[0] + coeffs[i,1]*psi[1])
    
    return psi_prime, E1, coeffs


def modified_energies(E):
    """Returns the eigenenergies of the modified unperturbed Hamiltonian. In this
    modified Hamiltonian, the quasi degenerate states (ground and first states)
    are truly degnerate, with value midway between the two energies.
    
    Input
        E : np.array
            Eigenenergies of original unperturbed Hamiltonian.
        
    Output
        E_prime : np.array
            Eigenergies of modified unperturbed Hamiltonian.
    """
    
    E_prime = copy.copy(E)
    
    # Replace quasi degenerate values with degenerate value midway between
    E_prime[0] = E_prime[1] = 0.5*(E[0]+E[1])
    
    return E_prime


def modified_perturbation_unprimed(H1, E_prime, E10, order):
    """Returns the modified perturbation in the unmodified basis that spans the
    quasi degenerate subspace, which is presumed to be the ground and first 
    excited states.
    
    Input
        H1 : np.array
            Original perturbation. Should be of the form H1 = -e * E_field * xx
        E10 : np.float
            Energy difference between ground and first excited state.
        order : int
            desired order of perturbation theory. Only goes up to 2 at the moment.
    Output 
        xx_prime : np.array
            The modified perturbation
    """
    if order > 2:
        assert False; "This order of perturbation theory is not implemented."
    if order == 1:
        return np.array([[H1[0,0] - 0.5*E10, H1[0,1]],[H1[1,0], H1[1,1] + 0.5*E10]])
    elif order == 2:
        return np.array([[H1[0,0] - 0.5 * E10 - H1[0,2:].dot(H1[2:,0] / (E_prime[2:] - E_prime[0])), H1[0,1]- H1[0,2:].dot(H1[2:,1] / (E_prime[2:] - E_prime[0]))],[H1[1,0]- H1[1,2:].dot(H1[2:,0] / (E_prime[2:] - E_prime[0])), H1[1,1] + 0.5*E10- H1[1,2:].dot(H1[2:,1] / (E_prime[2:] - E_prime[0]))]])

def modified_perturbation_primed(xx_prime, x, psi, psi_prime, E10, units):
    """Returns the modified perturbation in the modified basis. In this basis
    the ground and first excited states are fully degenerate but have no overlap
    via the perturbation.
    
    Input
        xx_prime : np.array
            Modified position matrix
        x : np.array
            Spatial grid
        psi : np.array
            Original unperturbed wavefunctions.
        psi_prime : np.array
            Unperturbed wavefunctions that diagonalize the modified perturbation.
        E10 : float
            Energy difference between quasi degenerate states (ground and first).
        units: class
            class whose attributes are the fundamental constants e, m, c, etc.
            
    Output :
        H1_tilde_primed : np.array
            Modified perturbation in primed basis.
    """
    H1_tilde_primed = copy.copy(xx_prime)
    
    # The diagonal elements:
    for i in range(2):
        H1_tilde_primed[i,i] = (-units.e * xx_prime[i,i] 
        + 0.5*E10*project(x, psi_prime[i], psi[1])*project(x, psi[1], psi_prime[i]) 
        - 0.5*E10*project(x,psi_prime[i], psi[0])*project(x,psi[0], psi_prime[i]))
        
    # The transition moment between the quasi degenerate states:
    H1_tilde_primed[1,0] = (-units.e * xx_prime[1,0] 
    + 0.5*E10*project(x, psi_prime[1], psi[1])*project(x, psi[1], psi_prime[0]) 
    - 0.5*E10*project(x,psi_prime[1], psi[0])*project(x,psi[0], psi_prime[0]))

    H1_tilde_primed[0,1] = (-units.e * xx_prime[0,1] 
    + 0.5*E10*project(x, psi_prime[0], psi[1])*project(x, psi[1], psi_prime[1]) 
    - 0.5*E10*project(x,psi_prime[0], psi[0])*project(x,psi[0], psi_prime[1]))

    # We assert that the primed basis diagonalizes the modified perturbation
    #   in the quasi degenerate subspace
    assert np.allclose([H1_tilde_primed[1,0], H1_tilde_primed[0,1]], [0,0]), "Primed states do not diagonalize modified perturbation."    
    
    return H1_tilde_primed

def lift_degen(E, x, xx, psi, units, order):
    """Returns the set of eigenstates that diagonalize the modified perturbation, 
    the corresponding coefficients of the linear combination, the first order 
    energy corrections to the modified states, and the position matrix in the
    modified basis.
    
    Input
        E : np.array
            unperturbed eigenenergies. E0 and E1 are treated as quasidegenerate.
        x : np.array
            spatial grid
        xx : np.array
            posiiton matrix in original unperturbed basis
        units : class
            fundamental constants
        
    Output
        alpha : complex
            polarizability
    """
    # The first two states are degenerate for the 
    #   modified unperturbed Hamiltonian, so we modify the first two elements of E
    #   to get the modified unperturbed energies E_prime:
    E_prime = modified_energies(E)
    
    # Compute modified perturbation in original basis.
    V_tilde_unprimed = modified_perturbation_unprimed(-units.e * xx, E_prime, E[1]-E[0], order)
    
    # Compute the linear combination of degenerate states that diagonalize the modified
    #   perturbation V_tilde_unprimed
    psi_prime, E1, coeffs = lin_comb(V_tilde_unprimed, x, psi)
    
    # Compute transition matrix in primed basis
    xx_prime = solver_utils.make_position_matrix(x, psi_prime)
    
    return psi_prime, coeffs, E_prime, xx_prime