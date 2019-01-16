# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 10:56:10 2019

@author: Ethan
"""

import numpy as np
import nlopy
from nlopy.quantum_solvers import solver_1D, solver_utils

from nlopy import utils
from nlopy.hyperpol.sum_over_states import sos_utils

x = utils.position_space(1,0.01)
V = 0*x
units = utils.Units('atomic')
psi, E = solver_1D.solver_1D(x, V, units)
xx = solver_utils.make_position_matrix(x, psi)

xx_ = xx - xx[0,0] * np.eye(len(E))
E_ = E - E[0]

w1 = 1
w2 = 2
w3 = 3

# term11
term11 = sos_utils.gamma_term11(xx_, E_, w1, w2, w3, units)
term11_ = 0
for n in range(1,15):
    for m in range(1,15):
        for l in range(1,15):
            term11_ += xx_[0,n] * xx_[n,m] * xx_[m,l] * xx_[l,0] / (E_[n] - w1 - w2 - w3) / (E_[m] - w1 - w2) / (E_[l] - w1)
            
assert np.allclose(term11, term11_), "--term11--"

# term12
term12 = sos_utils.gamma_term12(xx_, E_, w1, w2, w3, units)
term12_ = 0
for n in range(1,15):
    for m in range(1,15):
        for l in range(1,15):
            term12_ += xx_[0,n] * xx_[n,m] * xx_[m,l] * xx_[l,0] / (E_[n] + w1) / (E_[m] - w3 - w2) / (E_[l] - w3)
            
assert np.allclose(term12, term12_), "--term12--"

# term13
term13 = sos_utils.gamma_term13(xx_, E_, w1, w2, w3, units)
term13_ = 0
for n in range(1,15):
    for m in range(1,15):
        for l in range(1,15):
            term13_ += xx_[0,l] * xx_[l,m] * xx_[m,n] * xx_[n,0] / (E_[n] - w1) / (E_[m] + w3 + w2) / (E_[l] + w3)
            
assert np.allclose(term13, term13_), "--term13--"

# term14
term14 = sos_utils.gamma_term14(xx_, E_, w1, w2, w3, units)
term14_ = 0
for n in range(1,15):
    for m in range(1,15):
        for l in range(1,15):
            term14_ += xx_[0,l] * xx_[l,m] * xx_[m,n] * xx_[n,0] / (E_[n] + w1 + w2 + w3) / (E_[m] + w1 + w2) / (E_[l] + w2)
            
assert np.allclose(term14, term14_), "--term14--"

# term21
term21 = sos_utils.gamma_term21(xx_, E_, w1, w2, w3, units)
term21_ = 0
for n in range(1,15):
    for m in range(1,15):
        term21_ += xx_[0,n] * xx_[n,0] * xx_[0,m] * xx_[m,0] / (E_[n] - w1 - w2 - w3) / (E_[m] - w1) / (E_[n] - w3)
            
assert np.allclose(term21, term21_), "--term21--"

# term22
term22 = sos_utils.gamma_term22(xx_, E_, w1, w2, w3, units)
term22_ = 0
for n in range(1,15):
    for m in range(1,15):
        term22_ += xx_[0,n] * xx_[n,0] * xx_[0,m] * xx_[m,0] / (E_[n] - w3) / (E_[m] + w2) / (E_[n] - w1)
            
assert np.allclose(term22, term22_), "--term22--"

# term23
term23 = sos_utils.gamma_term23(xx_, E_, w1, w2, w3, units)
term23_ = 0
for n in range(1,15):
    for m in range(1,15):
        term23_ += xx_[0,n] * xx_[n,0] * xx_[0,m] * xx_[m,0] / (E_[n] + w1 + w2 + w3) / (E_[m] + w1) / (E_[n] + w3)
            
assert np.allclose(term23, term23_), "--term23--"

# term24
term24 = sos_utils.gamma_term24(xx_, E_, w1, w2, w3, units)
term24_ = 0
for n in range(1,15):
    for m in range(1,15):
        term24_ += xx_[0,n] * xx_[n,0] * xx_[0,m] * xx_[m,0] / (E_[n] + w3) / (E_[m] + w1) / (E_[n] - w2)
            
assert np.allclose(term24, term24_), "--term24--"

# term31
term31 = sos_utils.gamma_term31(xx_, E_, w1, w2, w3, units)
term31_ = 0
for n in range(1,15):
    term31_ += xx_[0,n] * xx_[n,0] / (E_[n] - w1 - w2 - w3) / (E_[n] - w1 - w2) / w1 / w2
            
assert np.allclose(term31, term31_), "--term31--"

# term32
term32 = sos_utils.gamma_term32(xx_, E_, w1, w2, w3, units)
term32_ = 0
for n in range(1,15):
    term32_ += xx_[0,n] * xx_[n,0] / (E_[n] + w1 + w2 + w3) / (E_[n] + w1 + w2) / w1 / w2
            
assert np.allclose(term32, term32_), "--term32--"

# term33
term33 = sos_utils.gamma_term33(xx_, E_, w1, w2, w3, units)
term33_ = 0
for n in range(1,15):
    term33_ += xx_[0,n] * xx_[n,0] / (E_[n] + w1) / (E_[n] - w3 - w2) / w3 / w2
            
assert np.allclose(term33, term33_), "--term33--"

# term34
term34 = sos_utils.gamma_term34(xx_, E_, w1, w2, w3, units)
term34_ = 0
for n in range(1,15):
    term34_ += xx_[0,n] * xx_[n,0] / (E_[n] - w1) / (E_[n] + w3 + w2) / w3 / w2
            
assert np.allclose(term34, term34_), "--term34--"