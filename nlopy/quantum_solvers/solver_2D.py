import scipy as sp
from scipy import linalg
import scipy.linalg
import numpy as np
from concurrent import futures
from nlopy.quantum_solvers import solver_utils

def solver_2D(x, y, V, units, num_states=15):
    """Uses finite difference to discretize and solve for the eigenstates and 
    energy eigenvalues two dimensional potentials.
    
    Assumes infinite walls at both ends of the problem space.
    
    Input
        x : np.array([x_i]) 
            The spacial grid points including the endpoints
        V : np.array([V(x_i)]) 
            The potential function defined on the grid
        units : class
            Class whose attributes are the fundamental constants hbar, e, m, c, etc.
        
    Output
        psi : np.array([psi_0(x), ..., psi_N(x)]) where N = NumStates
            Eigenfunctions of Hamiltonian
        E : np.array([E_0, ..., E_N]) 
            Eigenvalues of Hamiltonian
        
    Optional:
        num_states : int, default 15
            Dictates the number of states to solve for. Must be less than
            the number of spatial points - 2.
    """