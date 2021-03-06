# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 12:33:36 2016

@author: Sean Mossman
"""

import numpy as np
from matplotlib import pyplot as plt
#import os
#from scipy.integrate import simps


fsize = 10
params = {
    'figure.figsize': [3.35,2.0],
    #'figure.figsize': [9.83,7.5],
    'figure.dpi': 300,
    'font.size': fsize,
    'font.family' : 'serif',
    'font.weight' : 'light',
    'axes.labelsize': fsize,
    'legend.fontsize': fsize,
    'legend.numpoints': 1,
    'legend.handlelength': 1,
    'xtick.labelsize': fsize,
    'ytick.labelsize': fsize,
    'axes.linewidth' : 0.5,
    'figure.autolayout': False,
    'figure.subplot.left'    : 0.18,  #controlling whitespace
    'figure.subplot.right'   : 0.95,    
    'figure.subplot.bottom'  : 0.15,    
    'figure.subplot.top'     : 0.95,    
    'figure.subplot.wspace'  : 0.3,    # the amount of width reserved for blank space between subplots
    'figure.subplot.hspace'  : 0.3,   # the amount of height reserved for white space between subplots
    'savefig.dpi' : 200,
    #'savefig.format' : 'pdf',
    'savefig.format' : 'png',
    'text.usetex': True
}
plt.rcParams.update(params)
