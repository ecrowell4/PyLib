"""This file containts funtions and classes that should be of general use in Nonlinear optics, regardless
of application."""

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

	x = np.arange((L + dx) / dx) * dx

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
		elif unit_type is 'Gaussian':
			self.hbar = 6.58211928e-16 #eV.s
			self.e = 4.8032042e-10 #esu
			self.c = 2.99792458e10 #cm/s
			self.m = 0.51099890e6 / (self.c)**2 #eV.s^2/cm^2 
		else:
			assert False, "Indicated units not supported. Please choose from ['atomic', Gaussian']."
