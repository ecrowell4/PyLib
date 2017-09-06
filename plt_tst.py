# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:35:16 2017

@author: Owner
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_(x, func):
    plt.figure()
    plt.plot(x, func)
    plt.savefig('tst.jpeg')