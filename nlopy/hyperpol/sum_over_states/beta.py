# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 11:21:17 2017

@author: Ethan Crowell
"""

import numpy as np
from nlopy.sum_over_states import sos_utils as sos


def beta_eee(E, xx, omega, intrinsic=False):
	"""Returns the diagonal element of the first hyperpolarizability.

	Input
		E : np.array
			array of ordered eigenenergies. Damping is included by letting this array
			have complex entries.
		xx : np.array
			transition matrix
		omega : list
			incident field frequencies [omega1, omega2]
		intrinsic : bool
			if True, then xx must be normalized by xmax before input
			and E must be normalized by E[1] before input.

	Output
		beta : complex
			the first hyperpolarizability.
	"""

	# assert consistent dimensions
	assert((len(E)==len(xx[0])), "dimensions of E and xx do not match."

	# determine number of eigenstates to be used in computing beta
	num_states = len(E)

    #Take all mu -> bar{mu}
    xx = xx - xx[0,0] * np.eye(num_states)
    
    #Take all E -> E - E[0]
    E = E - E[0]
            
    #Calculate beta
    beta = 0.5 * e**3 * ((xx[0,1:] * sos_utils.D2(E[1:], omega[0], omega[1], units)).dot(xx[1:,1:].dot(xx[1:,0] * sos.D1(E[1:], omega[0], units)))
    + (xx[0,1:] * sos.D2(E[1:], omega[1], omega[0], units)).dot(xx[1:,1:].dot(xx[1:,0] * sos.D1(E[1:], omega[1], units)))
    + (xx[0,1:] * sos.D1(E[1:], omega[1], units)).dot(xx[1:,1:].dot(xx[1:,0] * sos.D1(E[1:].conjugate(), -omega[0], units)))
    + (xx[0,1:] * sos.D1(E[1:], omega[0], units)).dot(xx[1:,1:].dot(xx[1:,0] * sos.D1(E[1:].conjugate(), -omega[1], units)))
    + (xx[0,1:] * sos.D2(E[1:].conjugate(), -omega[0], -omega[1], units)).dot(xx[1:,1:].dot(xx[1:,0] * sos.D1(E[1:].conjugate(), -omega[0], units)))
    + (xx[0,1:] * sos.D2(E[1:].conjugate(), -omega[1], -omega[0], units)).dot(xx[1:,1:].dot(xx[1:,0] * sos.D1(E[1:].conjugate(), -omega[1], units))))
    
    if intrinsic is True:
    	# normalize by constant (this assumes E and xx have been entered as E/E[1] and xx/xmax)
        return beta * (3/4)**(3/4) / 3
    else:
    	# return the actual value of beta.
        return beta