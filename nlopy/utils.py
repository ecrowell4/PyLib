
"""This file containts funtions and classes that should be of general use in Nonlinear optics, regardless
of application."""

import numpy as np
from numba import jit

def position_space(L, N, centered=False, periodic=False):
    """Returns an array that represents a discretized representation of
    the interval [L_min, L_min + L], enpoints included. 

    Input
        L : float
            Length of position space
        N : int
            number of points
        centered : bool
            if True, origin is at center of domain. Otherwise, origin 
            is at left.
        periodic : bool
            True if domain is periodic, in which case endpoint is excluded
            to avoid redundancy.

    Output
        x : np.array
            Spatial grid
    """

    dx = L / (N-1)

    if centered is True:
        left_adjust = L / 2
    else:
        left_adjust = 0

    if periodic is True:
        x = np.arange(N - 1) * dx - left_adjust
    else:
        x = np.arange(N) * dx - left_adjust

    return x
    
class Units():
    """Class whose attributes are the fundamental constants. User can choose from
    ['atomic', 'Gaussian'].
    Input
        unit_type : string
            Specifies the desired type of units
    """

    def __init__(self, unit_type):
        if unit_type is 'atomic':
            self.hbar = 1
            self.e = 1
            self.m= 1
            self.c = 137.036
            self.g = self.e / 2 / self.m / self.c  #gyromagnetic ratio
        elif unit_type is 'Gaussian':
            self.hbar = 6.58211928e-16 #eV.s
            self.e = 4.8032042e-10 #esu
            self.c = 2.99792458e10 #cm/s
            self.m = 0.51099890e6 / (self.c)**2 #eV.s^2/cm^2 
            self.g = self.e / 2 / self.m / self.c  #gyromagnetic ratio
        else:
            assert False, "Indicated units not supported. Please choose from ['atomic', Gaussian']."
            
def smooth_func(x, n, periodic=False):
    """ Generate smooth function by selecting n random Fourier coefficients.
    The function is assumed
    
    Input
        x : np.array
            Our spatial interval
        n : int
            desired number of Fourier terms
            
    Output
        f : np.array
            values of the generated function on x
            
    Optional
        periodic : bool
            if false, then a linear offset is added to break symmetry
    """
    
    # Length of interval
    L = x[len(x)-1] - x[0]

    # Create an array to store our function
    f = np.zeros(len(x))
    
    # Array of Fourier coeffs
    c1 = np.random.randn(n)
    c2 = np.random.randn(n)
    m = np.random.uniform(-10, 10)
    
    for i in range(n):
        f += c1[i] * np.sin(2*i*np.pi*x / L) + c2[i] * np.cos(2*i*np.pi*x / L)
    
    # We add a linear portion to break the symmetry
    if periodic == False:
        f += m * x

    return f

@jit(nopython=True)
def my_simps(f: complex, x: float, N: int)->complex:
    """Returns the integral of f over the domain x using the simposons
    rule. It is important to note that simpsons rule requires an even
    number of intervals. However, we typpically choose and odd number
    (even number of points). Thus, we use simpsons rule for all but the
    last segment and use a trapezoidal on the last segment. Then we do
    simpsons on all but first and use trapz on first. We then average
    the two results.    

    Input
        f : np.array
            function to be integrated
        x : np.aray
            spatial domain
        N : int
            len(x)

    Output
        int(f) : np.float
            integral of f over domain x
    """
    
    dx : float = x[1] - x[0]

    if N%2==0:
        result_left = f[0]
        for j in range(1, N-2, 2):
            result_left = result_left + 4 * f[j]
        for i in range(2, N-3, 2):
            result_left = result_left + 2 * f[i]
        result_left = result_left + f[-2]
        result_left = result_left * dx / 3
        
        result_left = result_left + (f[-2] + f[-1]) * dx / 2
        
        result_right = f[1]
        for j in range(2, N-1, 2):
            result_right = result_right + 4 * f[j]
        for i in range(3, N-2, 2):
            result_right = result_right + 2 * f[i]
        result_right = result_right + f[-1]
        result_right = result_right * dx / 3
        
        result_right = result_right + (f[0] + f[1]) * dx / 2
        return 0.5 * (result_left + result_right)
    else:
        result = f[0] + f[-1]
        for j in range(1, N, 2):
            result = result + 4 * f[j]
        for i in range(2, N-1, 2):
            result = result + 2 * f[i]
        return result * dx / 3
