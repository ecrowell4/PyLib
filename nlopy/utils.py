
"""This file containts funtions and classes that should be of general use in Nonlinear optics, regardless
of application."""

import numpy as np

def position_space(L, dx, L_min=0):
	"""Returns an array that represents a discretized representation of
	the interval [L_min, L_min + L], enpoints included. 

	Input
		L : float
			Length of position space
		dx : float
			Grid spacing
		L_min : float, optional
			Starting point of grid 

	Output
		x : np.array
			Spatial grid
	"""

	x = np.arange((L + dx) / dx) * dx - L_min

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
