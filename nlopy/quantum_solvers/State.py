import numpy as np
import copy

import nlopy
from nlopy import utils


class State(object):
    """Class which stores all information regarding the
    electronic state of the system.

    Attributes
        L : float
            length of the domain
        Nx : int
            number of abscissa
        dx : float
            spacing between abscissa
        x : np.array(dtype=float)
            spatial array
        Nu (Nd) : ints
            number of electrons with spin up (down)
        lagrange : bool
            if True, uses lagrange multipliers to keep states
            orthonormal
        exchange : bool
            if True, includes exchange in the calculations.

    Methods
        get_copy()
            returns a deep copy of the class"""

    def __init__(self, psiu, psid, L, Nx, Uc, state, Field=0, unit_type='atomic',
    	         lagrange=True, centered=True):
 
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

        # Initialize spatial domain
        self.L = L
        self.Nx = Nx
        self.dx = self.L / (self.Nx - 1)
        self.x = np.arange(self.Nx) * self.dx
        self.centered = centered
        if self.centered is True:
            self.x -= self.L / 2
        self.Field = Field

        # Initialize Hartree Fock parameters
        self.psiu = psiu
        self.psid = psid
        self.Nu = len(psiu)
        self.Nd = len(psid)
        self.Uc = Uc
        self.state = state
        self.lagrange = lagrange

    def get_copy(self):
        """Returns a copy of the current state. We will need
        multiple copies of the state for the Runge Kutta method."""
        y = copy.deepcopy(self)
        return y





