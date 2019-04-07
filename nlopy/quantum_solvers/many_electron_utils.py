
import numpy as np
from numba import jit
import scipy
from scipy import integrate

import nlopy
from nlopy import utils
from nlopy.quantum_solvers import solver_1D, solver_utils

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

def many_electron_dipole(rho, dx, units):
    """Returns the many electron dipole moment.
    
    Input
        rho : np.array
            many electron density
        dx : np.array
            grid spacing
        units : Class
            fundamental constants
            
    Output
        mu : float
            many electron dipole moment
    """

    return -units.e * np.trapz(rho * x, dx=dx)

def braket(psia, psib, dx):
    """Projects psia onto psib: <psia|psib>.

    Input
        psia : np.array
            function to be projected
        psib : np.array
            function to be projected onto
        dx : np.float
            grid spacing

    Output
        proj : np.float
            projection of psia onto psib
    """

    return integrate.simps(psia.conjugate() * psib, dx=dx)

def norm2(psia, dx):
    return integrate.simps(abs(psia)**2, dx=dx)

def make_orthogonal(psi, dx):
    """Returns a set of orthonormal eigenvectors via QR decomposition.

    Input
        psi : np.array
            array whose rows are eigenvectors that aren't
            mutually orthogonal
        dx : np.float
            grid spacing. Used to make state normal in the usual sense

    Output
        psi : np.array
            array whose rows are orthonormal vectors
    """

    Q, R = np.linalg.qr(psi.transpose())

    return -Q.transpose() / np.sqrt(dx)

def gram_schmidt(psi, dx, units):
    """Takes in a set of basis functions and returns an orthonormal basis.

    Input
        x : np.array
            spatial array
        psi : np.array
            array whose elements are the basis functions
        units : Class
            fundamental constants

    Output
        psi_gm : np.array
            array of orthonormal basis functions
    """
    # Determine number of basis functions
    N = len(psi)

    # Initialize array to store new evctors
    psi_gm = np.zeros(psi.shape, dtype=complex)

    # First vector doesn't change
    psi_gm[0] = psi[0]

    # Loop through each function and orthogonalize
    for k in range(N):
        psi_gm[k] = psi[k]
        for j in range(k):
            psi_gm[k] -= (braket(psi[k], psi_gm[j], dx) / braket(psi_gm[j], psi_gm[j], dx)) * psi_gm[j]
        psi_gm[k] /= np.sqrt(braket(psi_gm[k], psi_gm[k], dx))
    return psi_gm

def overlap_matrix(psi, dx):
    """ Returns the overlap of all states contained in psi. For orthonormal
    states, this should be equal to the idential operator.

    Input
        psi : np.array
            collection of states
        ds : np.float
            grid spacing

    Output
        S : np.array
            overlap matrix <n|m>
    """
    Ne = len(psi)
    S = np.zeros((Ne, Ne))
    for i in range(Ne):
        for j in range(Ne):
            S[i,j] = braket(psi[i], psi[j], dx)
    return S
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
    
    # Determine grid spacing
    dx = x[1] - x[0]
    
    # Create array whose ij element is xi - xj
    Deltax = np.outer(x, np.ones(len(x))) - np.outer(np.ones(len(x)), x)
    
    # Compute corresponding 1D Coulomb interaction energies
    U_coul = -2 * np.pi * q * integrate.simps(rho_charge * abs(Deltax), dx=dx, axis=1)

    return U_coul     

def get_Jb_1D(x, psib, units):
    """Return the pairwise direct integral of Hartree-Fock theory. This is basically the
    Coulomb energy associated with the electron in the bth state. This is 
    defined as the integral Jb = 2pi * q * Int[|psi_b|^2 r12]
    
    Input
        x : np.array
            spatial array
        psib : np.array
            wavefunction
        q : np.float
            charge associated with particle
        units : Class
            fundamental constants
            
    Output
        J : np.array
            the direct integral as defined above.
    """

    # Compute charge density associated with psib
    rho_el = -units.e * abs(psib)**2
    
    # Compute direct integral
    Jb = get_1D_coulomb_int(x, -units.e, rho_el)
    
    return Jb

def get_Jbpsi_1D(x, psia, psib, units):
    """Returns the action of the pairwise direct integral on the state due 
    to single state psib.
    
    Input
        x : np.array
            spatial array
        psia : np.array
            wavefunction. state to be acted on
        psib : np.array
            wavefunction. state creating coulomb potential
    """
    
    # Compute direct integral
    Jb = get_Jb_1D(x, psib, units)
    
    return Jb * psia

def get_Kbpsi_1D(x, psia, psib, units):
    """Returns the action of the pairwise exchange operator on the state.
    
    Input
        x : np.array
            spatial array
        psia : np.array
            wavefunction. state to be acted on
        psib : np.array
            wavefunction. state giving rise to exchange operator
        units : Class
            fundamental constants
    
    Output
        Kb_psi : np.array
            action of the exchange operator on the state.
    """
    
    # Determine grid spacing
    dx = x[1]-x[0]
    
    # Compute array who's ij element is xi - xj
    Deltax = np.outer(x, np.ones(len(x))) - np.outer(np.ones(len(x)), x)
    
    # Evaluate integral
    Kb = integrate.simps(psib.conjugate() * abs(Deltax) * psia, dx=dx, axis=1)
    
    # Act on state psib
    Kb_psi = Kb * psib
    
    return Kb_psi

@jit(nopython=True)
def get_Kbpsi_1D_jit(x : float, psia : complex, psib : complex, N : int, e : float)->complex:
    """Returns the action of the pairwise exchange operator on the state. This is jit compiled.
    
    Input
        x : np.array
            spatial array
        psia : np.array
            wavefunction. state to be acted on
        psib : np.array
            wavefunction. state giving rise to exchange operator
        N : int
            length of position array
        e : float
            electric charge
    
    Output
        Kb_psi : np.array
            action of the exchange operator on the state.
    """
    K: complex = np.zeros(N) 
    for i in range(len(x)):
        f : complex = psib.conjugate() * np.abs(x - x[i]) * psia
        K[i] = -2*np.pi * simps_(f, x, len(x))        
    return K * psib

def direct_integral(x, psi, a, Ne, units):
    """Returns the direct integral from Hartree Fock theory in 1D.
    This is just the sum of the pairwise direct integrals.

    Input
        x : np.array
            spatial array
        psi : np.array
            states
        a : int
            state to be acted on
        Ne : int
            number of electrons
        units : Class
            fundamental constants

    Output
        J : np.array
            the direct integral
    """
    
    # Compute charge density that ath electron sees
    rho_el = -units.e * prob_density(psi, x, Ne, a)

    # Compute Coulomb interaction
    J = get_1D_coulomb_int(x, -units.e, rho_el)

    return J * psi[a]

def exchange_integral(x, psi, a, Ne, units):
    """Returns the exchange integral from Hartree Fock theory in 1D.
    This is just the sum of the pairwise exchange integrals.

    Input
        x : np.array
            spatial array
        psi : np.array
            states
        a : int
            state to be acted on
        Ne : int
            number of electrons
        units : Class
            fundamental constants

    Output
        J : np.array
            the direct integral
    """
    
    # Initialize array for memory
    K = np.zeros(len(x), dtype=complex)

    # Compute the direct integral
    for b in np.delete(range(Ne), a):
        K += get_Kbpsi_1D(x, psi[a], psi[b], units)

    return K

@jit(nopython=True)
def exchange_integral_jit(x : float, psi : complex, a : int, Ne : int, N : int, e : float)->complex:
    """Returns the exchange integral from Hartree Fock theory in 1D.
    This is just the sum of the pairwise exchange integrals.

    Input
        x : np.array
            spatial array
        psi : np.array
            states
        a : int
            state to be acted on
        Ne : int
            number of electrons
        N : int 
            length of position array
        e : float
            electron charge

    Output
        J : np.array
            the direct integral
    """
    K: complex = np.zeros(N)
    for b in range(Ne):
        if b != a:
            integral: complex = get_Kbpsi_1D_fast(x, psi[a], psi[b], len(x), e)
            K = K + integral
    return K

def apply_f(x, psia, psi, V_arr, a, Ne, units, lagrange=False, exchange=False):
    """Returns the action of the Hartree-Fock operator on the state psi[a]. The
    HF operator is s.t. 
        F psia = h psia + sum(2Jb - Kb, b not a) psia
    where h is just the kinetic energy plus external potential.
    
    Input
        x : np.array
            spatial array
        psia : np.array
            state to which F is applied. (technically redundant, but makes RK easier)
        psi : np.array
            set of states
        a : np.array
            state to be acted on
        Ne : np.int
            number of electrons
        units : Class
            fundamental constants
        lagrange : bool
            if True, use lagrange multipliers to keep states orthogonal
        exchange : bool
            if True, include exchange term.
    
    Output
        f_psi : np.array
            array resulting from acting on psi[a] with HF operator.
    """

    # Determine grid spacing
    dx = x[1] - x[0]
    
    
    fpsia = (solver_utils.apply_H(psia, x, V_arr, units) 
        + 2 * direct_integral(x, psi, a, Ne, units))

    if exchange==True:
        fpsia = fpsia - exchange_integral_jit(x, psi, a, Ne, len(x), units.e)
    
    if lagrange==False:
        return fpsia
    else:
        Fpsia = fpsia
        for b in range(Ne):
            Fpsia -= (braket(psi[b], fpsia, dx) / braket(psi[b], psi[b], dx)) * psi[b]
        return Fpsia

def get_HF_energy(x, psi, Varr, Ne, units, exchange=False):
    """Returns the Hartree Fock energt for Ne electrons.
    
    Input
        x : np.array
            spatial array
        psi : np.array
            psi[i] is the ith electron orbital
        Varr : np.array
            Array of external potential energy
        Ne : np.int
            number of electrons
        units : Class
            fundamental constants
        exchange : bool
            if True, include the exchange integral
    
    Output
        E : np.float
            HF energy
    """

    if exchange==True:
        coeff = 1
    else:
        coeff = 0

    # initialize energy to zero
    E = 0
    
    # Determine grid spacing
    dx = x[1] - x[0]

    for a in range(Ne):
        # For each particle, compute fpsi (note the factor of 2 in apply_H)
        fpsi = 2 * (solver_utils.apply_H(psi[a], x, Varr, units) 
            + direct_integral(x, psi, a, Ne, units))

        if exchange==True:
            fpsi -= exchange_integral_jit(x, psi, a, Ne, len(x), units.e)

        E += integrate.simps(psi[a].conjugate() * fpsi, dx=dx)
    
    assert np.allclose(E.imag, 0), "Energy is not real valued"
    if E > 1 / dx**2:
        print("Grid spacing too coarse")    
    return E

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
        
        percent_diff = np.max(abs(E - Eprev))
        count += 1

    
    return psi, E

def assert_iteration_lim(n, max_iter):
    """This is to prevent excessively long loops. Once a maximum number
    of iterations has been reached, the current states are saved and the
    loop is closed.

    Input
        n : int
            current iterations number
        max_iter
            max allowed iterations.

    Output
        None
    """

    if n > max_iter:
        print('Maximum number of iterations ('+str(max_iter)+') exceeded.')
        np.save('../data_files/hartree_full_box/hartree_'+str(Ne)+'_max_iter', Es)
        np.save('../data_files/hartree_full_box/hartree_'+str(Ne)+'_max_iter', psi) 
        quit()

def update_dt(dt, updown, delta=0.1):
    """Updates the time step

    Input
        dt : np.float
            current step size
        delta : np.float
            small number by which to change dt. The percent change
            in dt is given by 100 * delta. 
        updown : string
            Increase time step by factor delta if "increase"
            Decrease time step by factor delta if "decrease"

    Output
        dt : np.float
            updated time step
    """
    if updown=="increase":
        dt *= (1 + delta)
        print("Increasing time step - dt = "+str(dt))
    elif updown=="decrease":
        dt *= (1 - delta)
        print("Decreasing time step - dt = "+str(dt))
    return dt