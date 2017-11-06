
import numpy as np
import copy

def modified_perturbation_matrix(V, E10):
    """Returns the modified perturbation in the unmodified basis that spans the
    quasi degenerate subspace, which is presumed to be the ground and first 
    excited states.
    
    Input
        V : np.array
            Original perturbation. Should be of the form V = e * E_field * xx
        E10 : np.float
            Energy difference between ground and first excited state.
        
    Output 
        xx_prime : np.array
            The modified perturbation
    """
    
    return np.array([[V[0,0] - 0.5*E10, V[0,1]],[V[1,0], V[1,1]+0.5*E10]])


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


def lift_degen(V_prime, x, psi):
    """Returns a linear combinations of two degnerate states that diagonalizes
    the modified perturbation. V
    
    Input
        V_prime : np.array
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
    E1, coeffs = np.linalg.eig(V_prime)
    
    coeffs = coeffs.transpose()
    
    # Most of the states will be unaffected, so we initially copy the entire array
    psi_prime = copy.copy(psi)
    
    # The ground and first excited states must be modified in order to diagonalize
    #   the modified perturbation:
    for i in range(2):
        # Create linear combinations
        psi_prime[i] = coeffs[i,0]*psi[0] + coeffs[i,1]*psi[1]
        # Normalize the resulting wavefunctions
        psi_prime[i] /= np.sqrt(np.trapz(psi_prime[i].conjugate() * psi_prime[i], x))
    
    return psi_prime, E1


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


def modified_position_matrix(xx, x, psi_prime, E10):
    """Returns position matrix elements using the modified wavefunctions from
    quasi degenerate perturbation theory. Note: uses the project function
    defined above.
    
    Input
        xx : np.array
            origin position matrix
        psi : np.array
            original unperturbed wavefunctions
        psi_prime : np.array
            Unperturbed wavefunctions that diagonalize the modified perturbation.
        E10 : float
            Energy difference between quasi degenerate states (ground and first).
    
    Output :
        xx_prime : np.array
            Modified position matrix
    """

    num_states = len(xx[0,:])
    xx_prime = copy.copy(xx)
    
    # The only elements that are affected are those that couple states to the 
    #   ground or first excited states. The perturbation is the same when one of
    #   the involved states is not one of the quasi degenerate states.
    for i in range(num_states):
        xx_prime[i,0] = np.trapz(psi_prime[i].conjugate() * x * psi_prime[0], x)
        xx_prime[i,1] = np.trapz(psi_prime[i].conjugate() * x * psi_prime[1], x)
    
    # When both states are from the quasi degenerate subspace, we have to use the
    #   modified perturbation. The diagonal elements:
    for i in range(2):
        xx_prime[i,i] = (np.trapz(psi_prime[0].conjugate() * x * psi_prime[0], x)
        - (-1)**i * 0.5*E10)    
    
    # The matrix must still be hermitian, so we accordingly assign the transpose
    #   elements
    xx_prime[0,:] = xx_prime[:,0].conjugate()
    xx_prime[1,:] = xx_prime[:,1].conjugate()
    
    return xx_prime


def modified_perturbation(xx_prime, x, psi, psi_prime, E10):
    """Returns the modified perturbation in the modified basis. In this basis
    the ground and first excited states are fully degenerate but have no overlap
    via the perturbation.
    
    Input
        xx_prime : np.array
            Modified position matrix
        psi_prime : np.array
            Unperturbed wavefunctions that diagonalize the modified perturbation.
        E10 : float
            Energy difference between quasi degenerate states (ground and first).
    
    Output :
        xx_prime_pert : np.array
            Modified position matrix
    """
    xx_prime_pert = copy.copy(xx_prime)
    
    # The transition moment between the quasi degenerate states:
    xx_prime_pert[1,0] = xx_prime_pert[0,1] = (xx_prime[1,0] 
    + 0.5*E10*project(x, psi_prime[1], psi[1])*project(x, psi[1], psi_prime[0]) 
    - 0.5*E10*project(x,psi_prime[1], psi[0])*project(x,psi[0], psi_prime[0]))
    
    return xx_prime_pert