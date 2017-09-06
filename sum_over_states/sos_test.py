# -*- coding: utf-8 -*-
"""
Created on Mon May  8 16:51:48 2017

@author: Owner
"""

from __future__ import division
import numpy as np
import sys
sys.path.append(r'C:\Users\Owner\Dropbox\PyLib')
sys.path.append('..')
import datetime as dt
from concurrent import futures
import time
from PyLib import *


def alpha(L, E, omega=0):
    e = 1
    m = 1
    c = 137.04
    gamma = e / 2 / m / c
    hbar = 1
    alpha = 0
    l = 1
    while l < len(E):
        alpha += L[0,l]*L[l,0] / (E[l]-E[0]-omega) + L[0,l]*L[l,0] / (E[l]-E[0]+omega) 
        l += 1
    return (gamma**2) * alpha