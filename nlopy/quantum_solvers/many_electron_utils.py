
import numpy as np
import nlopy
from nlopy import utils
from nlopy.quantum_solvers import solver_1D, solver_utils

import matplotlib.pyplot as plt

#==============================================================================
# General many electron utilities
#==============================================================================
def prob_density(psi, x, Ne, i):
    """Compute multielectron density that ith particle sees. If i is none, then
    computes total multielectron density.
    
    Input
        psi : np.array
            single particle wavefunctions
        x : np.array
            position space
        Ne : np.int
            number of electrons
        i : np.int
            label of electron. If None, then total probability density is computed.
    
    Output
        rho : np.array
            the probability density that the ith electron sees.
    """
    
    if i == None:
        # Compute total probability density
        rho = np.sum(abs(psi[:Ne])**2, axis=0)
    else:
        # Compute probability density excluding ith particle
        rho = np.sum(abs(psi[:i])**2, axis=0) + np.sum(abs(psi[i+1:Ne])**2, axis=0)
        
    return rho

def many_electron_dipole(rho, x, units):
	"""Returns the many electron dipole moment.
    
	Input
	    rho : np.array
	    	many electron density
        x : np.array
            position space
        units : Class
            fundamental constants
            
	Output
	    mu : float
	        many electron dipole moment
	"""

	return -units.e * np.trapz(rho * x, x)

#==============================================================================
# Utilities specific to Hartree Method
#==============================================================================
def get_1D_coulomb_int(x, q, rho_charge):
    """Compute the 1D coulomb interaction energy between charge q and charge
    density rho_charge. Note that this is in gaussian units.
    
    Input
        x : np.array
            position space
        q : np.float
            test charge
        rho_charge : np.array
            charge density that
    
    Output
        U : np.float
            interaction energy for ith electron.
    """
    
    # Create array whose ij element is xi - xj
    Deltax = np.outer(x, np.ones(len(x))) - np.outer(np.ones(len(x)), x)
    
    # Compute corresponding 1D Coulomb interaction energies
    U_coul = -2 * np.pi * q * np.trapz(rho_charge * abs(Deltax), x, axis=1)

    return U_coul             

def get_next_psi(psi_current, x, V, Ne, units):
    """For each state, determine the state corresponding to the next iteration.
    
    Input
        psi_current : np.array
            wavefunctions from current iteration
        x : np.array
            position array
        V : np.array
            external potential
        Ne : np.float
            number of electrons
        units : Class
            fundamental constants
    
    Output
        psi_next : np.array
            wavefunction for next iteration
        E_next : np.array
            energies associated with each electronic state
    """

    # Initialize arrays that will store results after iteration
    #psi_next = np.zeros((Ne, len(x)))
    psi_next = psi_current
    E_next = np.zeros(Ne)
    
    # For each electron, solve Hartree SE
    for i in range(Ne):

    	# Generate charge density that electron i sees due to other electrons
        #rho_i = -units.e * prob_density(psi_current, x, Ne, i)
        rho_i = -units.e * prob_density((psi_next+psi_current)/2, x, Ne, i)
        
        # Generate effective interaction energy
        U_i = get_1D_coulomb_int(x, -units.e, rho_i)
        
        # Solve corresponding Hartree equation
        temp_psi, temp_Es = solver_1D.solver_1D(x, V + U_i, units, num_states=Ne)
        
        psi_next[i] = temp_psi[i]
        E_next[i] = temp_Es[i]
        
    return psi_next, E_next

def get_hartree_states(psi0, E0, x, V, Ne, etol, units):
    """Compute the single electron states whose direct product forms the 
    many electron Hartree state.
    
    Input
        psi0 : np.array
            noninteracting single particle states
        E0 : np.array
            noninteracting single particle eigenenergies
        x : np.array
            position space
        V : np.array
            external potential
        Ne : np.int
            number of electrons
        etol : np.float
            convergence criterian for hartree method. Iterations will cease
            when percent difference in energy between iterations is less than
            this.
        units : Class
            fundamental constants
    
    Output
        psi : np.array
            single electrons states whose direct product is the many electron
            state.
        E : np.array
            single electron energies whose sum is the many electron energy
    """
    
    # initialize some value for the percent difference
    percent_diff = 1
    
    # Create count variable that will count number of iterations
    count = 0
    
    # Loop needs previous energy, so must define this for first loop
    psi = psi0[:Ne]
    E = E0[:Ne]
    
    
    
    # While percent diff is bigger than tolerance, continue with iterations:
    flag = False
    while flag == False:
    #while percent_diff > etol:
        print(count)
        Eprev = E
        psi, E = get_next_psi(psi, x, V, Ne, units)
        percent_diff = abs(np.sum(E) - np.sum(Eprev))
        
        if np.allclose(percent_diff, 0, etol) == True:
            flag = True
        
        plt.figure()
        plt.title(np.sum(E))
        plt.plot(x, psi[0], label='grnd state after '+str(count+1)+' iterations')
        plt.plot(x, psi[1], label='frst exc stt')
        plt.plot(x, psi[2], label='scnd exc st')
        plt.legend()
        plt.xlabel('position')
        plt.ylabel(r'$|\psi|^2$')
        plt.savefig('../data_files/'+str(count)+'png')
        plt.close()
#        
        percent_diff = np.max(abs(E - Eprev))
        count += 1

    
    return psi, E