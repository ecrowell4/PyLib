# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 14:49:11 2017

@author: Owner
"""

import numpy as np
# Import my personal library.
import sys
sys.path.append(r'C:\Users\Owner\Dropbox\PyLib')    # path for my machine
sys.path.append('..')   # path for Kamiak
import PyLib as pl

N = 1000
L = 1
num_states = 30
#==============================================================================
# Constants and parameters:
e = 1
m = 1 
c = 137.036
hbar = 1
x = np.linspace(0,L,N)
dx = x[1] - x[0]

V = 10*np.sin(np.pi * x)
psi, E, xx = pl.SEsolvers.SEsolve(x, V, 30)
alphas = pl.SOS_alpha.alpha_ee(E, xx, omega=0.5*(E[1]-E[0]), damping=True)
