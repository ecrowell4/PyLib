class Units():
	def __init__(self, unit_type):
		if unit_type is 'atomic':
			self.hbar = 1
			self.e = 1
			self.m= 1
			self.c = 137.036
		elif unit_type is 'Gaussian':
			self.hbar = 6.58211928e-16 #eV.s
			self.c = 2.99792458e10 #cm/s
			self.m = 0.51099891e6 / (self.c)**2 #eV.s^2/cm^2 
		else:
			assert False, "Indicated units not supported. Please choose from ['atomic', Gaussian']."