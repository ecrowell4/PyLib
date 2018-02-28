# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 11:21:17 2017

@author: Ethan Crowell
"""

import numpy as np
from nlopy.hyperpol.sum_over_states import sos_utils


def beta_eee(E, xx, units, omega=[0,0], intrinsic=False, n=0):
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
        n : int
            state system is assumed to be in (i.e. n=0 -> ground state)

    Output
        beta : complex
            the first hyperpolarizability.
    """

    # assert consistent dimensions
    assert len(E)==len(xx[0]), "dimensions of E and xx do not match."

    # determine number of eigenstates to be used in computing beta
    num_states = len(E)

    #Take all mu -> bar{mu}
    xx = xx - xx[0,0] * np.eye(num_states)

    #Take all Em -> Emn
    E = E - E[n]
            
    #Calculate beta
    beta = 0.5 * units.e**3 * (
        (np.delete(xx[n,:], n) * sos_utils.D2(np.delete(E, n), omega[0], omega[1], units)).dot(np.delete(np.delete(xx, n, 0), n, 1).dot(np.delete(xx[:,n], n) 
    * sos_utils.D1(np.delete(E, n), omega[0], units)))
    + (np.delete(xx[n,:], n) * sos_utils.D2(np.delete(E, n), omega[1], omega[0], units)).dot(np.delete(np.delete(xx, n, 0), n, 1).dot(xx[1:,0] * sos_utils.D1(np.delete(E, n), omega[1], units)))
    + (np.delete(xx[n,:], n) * sos_utils.D1(np.delete(E, n), omega[1], units)).dot(np.delete(np.delete(xx, n, 0), n, 1).dot(xx[1:,0] * sos_utils.D1(np.delete(E, n).conjugate(), -omega[0], units)))
    + (np.delete(xx[n,:], n) * sos_utils.D1(np.delete(E, n), omega[0], units)).dot(np.delete(np.delete(xx, n, 0), n, 1).dot(xx[1:,0] * sos_utils.D1(np.delete(E, n).conjugate(), -omega[1], units)))
    + (np.delete(xx[n,:], n) * sos_utils.D2(np.delete(E, n).conjugate(), -omega[0], -omega[1], units)).dot(np.delete(np.delete(xx, n, 0), n, 1).dot(xx[1:,0] * sos_utils.D1(np.delete(E, n).conjugate(), -omega[0], units)))
    + (np.delete(xx[n,:], n) * sos_utils.D2(np.delete(E, n).conjugate(), -omega[1], -omega[0], units)).dot(np.delete(np.delete(xx, n, 0), n, 1).dot(xx[1:,0] * sos_utils.D1(np.delete(E, n).conjugate(), -omega[1], units)))
    )
    
    if intrinsic is True:
        # normalize by constant (this assumes E and xx have been entered as E/E[1] and xx/xmax)
        return beta * (3/4)**(3/4) / 3
    else:
        # return the actual value of beta.
        return beta