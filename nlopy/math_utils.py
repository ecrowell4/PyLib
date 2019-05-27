import numpy as np
from numba import jit

@jit(nopython=True)
def laplacian(y:complex, dx:float)->complex:
	"""Returns laplacian of complex valued function `y(x)`
	Dirichlet b.c.s are assumed, implying laplacian vanishes
	at the boundary."""
	ddf:complex = np.zeros(len(f)) + 0j
	ddf[1:-1] = (f[2:] - 2*f[1:-1] + f[:-2]) / dx**2
	return ddf


@jit(nopython=True)
def my_convolve(y:complex, h:complex, dx:float)->complex:
	"""Returns the convolution of two complex valued functions
	y*h."""

@jit(nopython=True)
def subtract_lagrange(f:complex, y:complex, dx:float)->complex:
	"""Subtacts off the set of lagrange multipliers Y from y.

	Input
	    f : np.array
	        the variation of some functional F.
	        In context of HF theory, F is action of HF operator on
	        som orbital
	    y : np.array
	        set of orbitals

	Output
	    F : np.array
	        the variation of functional with constraints imposed.
	"""

