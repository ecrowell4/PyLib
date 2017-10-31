# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 21:23:02 2017

@author: Ethan Crowell

Contains utility functions for solving the Schrodinger equation.
"""

import numpy as np
import scipy as sp
from numba import jit

def kinetic_energy(dx, N, units, boundary):
    """Construct finite difference approximation for kinetic energy operator.
    
    Input
        dx : float
            spatial grid spacing
        N : int
            number of points representing posiition space
        units : class
            fundamental constants
        boundary : string
            describes boundary conditions. Options are
                hard_wall
                periodic
    
    Output
        Top : np.array((N-2, N-2))
            finite difference representation of kinetic energy operator in position 
            space. The endpoints of the space are left out, as they are defined 
            by the boundary conditions.
    """
    
    # Define N_tilde, which will be the dimension of our operator. 
    if boundary is 'hard_wall':
        # For hard walls, the matrix equation determines all but the two enpoints.
        N_tilde = N-2
    if boundary is 'periodic':
        # For periodic boundaris, the matrix equation determines all but one endpoint.
        N_tilde = N-1

    # Create finite difference approximation of second derivative operator
    d2_psi = -2 * np.eye(N_tilde) / dx**2 + np.eye(N_tilde, k=1) / dx**2
            + np.eye(N_tilde, k=-1) / dx**2

    if boundary is 'periodic':
        # This condition forces the derivative of the function to be periodic as well.
        H[0, N_tilde-1] = -1 / 2 / dx**2
        H[N_tilde-1, 0] = -1 / 2 / dx**2 
     
    # Convert the second derivative to units of energy:
    Top = -(units.hbar**2 / 2 / units.m) * d2_psi
    
    return Top


def potential_energy(V, boundary):
    """Construct finite difference approximation for kinetic energy operator.
    
    Input
        V : np.array(N)
            array represention of potential energy function
        boundary : string
            describes boundary conditions. Options are
                hard_wall
                periodic
    
    Output
        Vop : np.array
            representation of potential energy operator in position space.
            For h
    """
    
    # Matrix representing potential energy is diagonal in position space
    if boundary=='hard_wall':
        Vop = np.diag(V[1:-1])
    elif boundary=='periodic':
        Vop = np.diag(V[1:])
    
    return Vop

    
def hamiltonian(dx, V, units, boundary='hard_wall'):
    """Construct Hamiltonian matrix using a finite difference scheme.
    
    Input
        dx : float
            grid spacing
        V : np.array(len(x))
            potential energy funciton
        units : class
            class whose attributes are the fundamental constants hbar, e, m, c.
        boundary : string
            describes boundary conditions. Options are
                hard_wall
                periodic
            
    Output
        H : np.array((len(x), len(x)))
    """
    
    # Assert that 'boundary' is one of the two valid options
    assert boundary is 'hard_wall' or boundary is 'periodic', "Invalid boundary condition specified."
    # Determine number of points representing position space:
    N = len(V)
    
    # Construct operator representations in position space:
    Top = kinetic_energy(dx, N, units, periodic)
    Vop = potential_energy(V, periodic)

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