# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 10:56:10 2019

@author: Ethan
"""

import numpy as np
import nlopy
from nlopy.quantum_solvers import solver_1D, solver_utils

from nlopy import utils
from nlopy.hyperpol.sum_over_states import sos_utils, alpha, gamma


#=============================================================================================
# We test sos_utils for particle in a 1D box. Note that such a particle has no angular momentum
# However, we can still compute the derivatives and make sure that the functions in sos_utils
# are computing the correct summations.
L = 1
Nx = 101
dx = L / (Nx - 1)
x = np.arange(Nx)*dx
V = 0*x
units = utils.Units('atomic')
psi, E = solver_1D.solver_1D(x, V, units)
xx = solver_utils.make_position_matrix(x, psi)
L = solver_utils.make_angular_momentum(x, psi, units)
w1 = 0
w2 = 0
w3 = 0

n_ = 0
xx_ = xx - xx[n_,n_] * np.eye(len(E))
L_ = L - L[n_,n_] * np.eye(len(E))
E_ = E - E[n_]
#==============================================================================================
# We test for alpha, gamma
alpha_ = alpha.alpha_ee(E_, xx_, units, w1, n=n_)
alpha_tst = 0
for i in range(0,len(E_)):
    if i!=n_:
        alpha_tst += units.e**2 * xx_[n_,i] * xx_[i,n_] * (1/(E_[i] - w1) + 1/(E_[i] + w1))
print(alpha_, alpha_tst)
assert np.allclose(alpha_, alpha_tst), "--alpha--"

gamma_ = gamma.gamma_eeee_(E_, xx_, units, np.array([w1, w2, w3]), n=n_)

#===============================================================================================
# We test term by term
# term11
#term11 = sos_utils.gamma_term11(xx_, L_, E_, w1, w2, w3, units, gamma_type='eeee', n=n_)
term11 = sos_utils.gamma_term11(xx_, xx_, xx_, xx_, E_, w1, w2, w3, units, n=n_)
term11_ = 0
for n in range(0,15):
    for m in range(0,15):
        for l in range(0,15):
            if n!=n_ and m!=n_ and l!=n_:
                term11_ += units.e**4 * units.g**0 * xx_[n_,n] * xx_[n,m] * xx_[m,l] * xx_[l,n_] / (E_[n] - w1 - w2 - w3) / (E_[m] - w1 - w2) / (E_[l] - w1)
assert np.allclose(term11, term11_), "--term11--"



# term12
term12 = sos_utils.gamma_term12(xx_, xx_, xx_, xx_, E_, w1, w2, w3, units, n=n_)
term12_ = 0
for n in range(0,15):
    for m in range(0,15):
        for l in range(0,15):
            if n!=n_ and m!=n_ and l!=n_:
                term12_ += units.e**4 * units.g**0 * xx_[n_,n] * xx_[n,m] * xx_[m,l] * xx_[l,n_] / (E_[n] + w1) / (E_[m] - w3 - w2) / (E_[l] - w3)
assert np.allclose(term12, term12_), "--term12--"

# term13
term13 = sos_utils.gamma_term13(xx_, xx_, xx_, xx_, E_, w1, w2, w3, units, n=n_)
term13_ = 0
for n in range(0,15):
    for m in range(0,15):
        for l in range(0,15):
            if n!=n_ and m!=n_ and l!=n_:
                term13_ += units.e**4 * units.g**0 * xx_[n_,l] * xx_[l,m] * xx_[m,n] * xx_[n,n_] / (E_[n] - w1) / (E_[m] + w3 + w2) / (E_[l] + w3)      
assert np.allclose(term13, term13_), "--term13--"

# term14
term14 = sos_utils.gamma_term14(xx_, xx_, xx_, xx_, E_, w1, w2, w3, units, n=n_)
term14_ = 0
for n in range(0,15):
    for m in range(0,15):
        for l in range(0,15):
            if n!=n_ and m!=n_ and l!=n_:
                term14_ += units.e**4 * units.g**0 * xx_[n_,l] * xx_[l,m] * xx_[m,n] * xx_[n,n_] / (E_[n] + w1 + w2 + w3) / (E_[m] + w1 + w2) / (E_[l] + w2)            
assert np.allclose(term14, term14_), "--term14--"

# term21
term21 = sos_utils.gamma_term21(xx_, xx_, xx_, xx_, E_, w1, w2, w3, units, n=n_)
term21_ = 0
for n in range(0,15):
    for m in range(0,15):
        if n!=n_ and m!=n_:
            term21_ += units.e**4 * units.g**0 * xx_[n_,n] * xx_[n,n_] * xx_[n_,m] * xx_[m,n_] / (E_[n] - w1 - w2 - w3) / (E_[m] - w1) / (E_[n] - w3)
assert np.allclose(term21, term21_), "--term21--"

# term22
term22 = sos_utils.gamma_term22(xx_, xx_, xx_, xx_, E_, w1, w2, w3, units, n=n_)
term22_ = 0
for n in range(0,15):
    for m in range(0,15):
        if n!=n_ and m!=n_:
            term22_ += units.e**4 * units.g**0 * xx_[n_,n] * xx_[n,n_] * xx_[n_,m] * xx_[m,n_] / (E_[n] - w3) / (E_[m] + w2) / (E_[n] - w1)
assert np.allclose(term22, term22_), "--term22--"

# term23
term23 = sos_utils.gamma_term23(xx_, xx_, xx_, xx_, E_, w1, w2, w3, units, n=n_)
term23_ = 0
for n in range(0,15):
    for m in range(0,15):
        if n!=n_ and m!=n_:
            term23_ += units.e**4 * units.g**0 * xx_[n_,n] * xx_[n,n_] * xx_[n_,m] * xx_[m,n_] / (E_[n] + w1 + w2 + w3) / (E_[m] + w1) / (E_[n] + w3)
assert np.allclose(term23, term23_), "--term23--"

# term24
term24 = sos_utils.gamma_term24(xx_, xx_, xx_, xx_, E_, w1, w2, w3, units, n=n_)
term24_ = 0
for n in range(0,15):
    for m in range(0,15):
        if n!=n_ and m!=n_:
            term24_ += units.e**4 * units.g**0 *xx_[n_,n] * xx_[n,n_] * xx_[n_,m] * xx_[m,n_] / (E_[n] + w3) / (E_[m] + w1) / (E_[n] - w2)
assert np.allclose(term24, term24_), "--term24--"


gamma_tst = term11_ + term12_ + term13_ + term14_ - term21_ - term22_ - term23_ - term24_
print(gamma_, gamma_tst)