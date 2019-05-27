import numpy as np
from numba import jit

@jit(nopython=True)
def laplacian(y:complex, dx:float)->complex:
	"""Returns laplacian of complex valued function `y(x)`
	Dirichlet b.c.s are assumed, implying laplacian vanishes
	at the boundary."""

@jit(nopython=True)
def my_convolve(y:complex, h:complex, dx:float)->complex:
	"""Returns the convolution of two complex valued functions
	y*h."""