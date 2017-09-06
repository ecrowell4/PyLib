# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 13:46:29 2017

@author: Owner
"""

import numpy as np
from scipy.optimize import curve_fit

def cycloid(x, x0, xM, n):
    return x0 * (1 - (x / xM)**(1/n))**n

def cycloid_fit(fit, values, bins):
    """This function returns a cycloid fit where
    fit : func
        desire function for fitting
    values : np.array
        raw data from monte carlo run
    bins : desired number of bins for historgram
    """
    
    bins, edges = np.histogram(dthetas, bins)
    
    centers = np.zeros(len(bins))
    for i in range(len(bins)):
        centers[i] = 0.5 * (edges[i] + edges[i+1])