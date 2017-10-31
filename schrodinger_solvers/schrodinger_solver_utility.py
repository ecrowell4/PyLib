# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 21:23:02 2017

@author: Ethan Crowell

Contains utility functions for solving the Schrodinger equation.
"""

import numpy as np
import scipy as sp
from numba import jit

def kinetic_energy(dx, N, units):
    """Construct finite difference approximation for kinetic energy operator.
    
    Input
        dx : float
            grid spacing
        N : int
            number of points representing posiition space
        units : class
            fundamental constants
    
    Output
        Top : np.array((N-2, N-2))
            finite difference representation of kinetic energy operator in position 
            space. The endpoints of the space are left out, as they are defined 
            by the boundary conditions.
    """
    
    # Create finite difference approximation of second derivative operator
    d2_psi = -2 * np.eye(N-2) / dx**2 + np.eye(N-2, k=1) / dx**2
            + np.eye(N-2, k=-1) / dx**2
     
    # The kinetic energy is the second derivative times some consants
    Top = -(units.hbar**2 / 2 / units.m) * d2_psi
    
    return Top


def potential_energy(V):
    """Construct finite difference approximation for kinetic energy operator.
    
    Input
        V : np.array(N)
            array represention of potential energy function
    
    Output
        Vop : np.array((N-2, N-2))
            representation of potential energy operator in position space. The
            endpoints of the space are left out, as they are defined by the boundary
            conditions.
    """
    
    # Matrix representing potential energy is diagonal in position space
    Vop = np.diag(V[1:-1])
    
    return Vop

    
def hamiltonian(dx, V, units):
    """Construct Hamiltonian matrix using a finite difference scheme.
    
    Input
        dx : float
            grid spacing
        V : np.array(len(x))
            potential energy funciton
        units : class
            class whose attributes are the fundamental constants hbar, e, m, c.
            
    Output
        H : np.array((len(x), len(x)))
    """
    
    # Determine number of points representing position space:
    N = len(V)
    
    # Construct operator representations in position space:
    Top = kinetic_energy(dx, N, units)
    Vop = potential_energy(V)

    return Top + Vop
 

def position_matrix(x, psi):
    """Construct the matrix elements of the position operator:
    
    Input
        x : np.array
            array representing position space
        psi : np.array((num_states, len(x)))
            array containing eigenfunctions of Hamiltonian
    
    Output
        xx : np.array((num_state, num_states))
    """
    
    # Determine number of states that are being used:
    num_states = len(psi[:])
    
    # Allocate memeory to store the values:
    xx = np.zeros((num_states, num_states))
    
    # Compute the matrix elements: 
    for i in range(num_states):
        for j in range(num_states):
            xx[i,j] = np.trapz(psi[i].conjugate() * x *psi[j], x)
            
    return xx