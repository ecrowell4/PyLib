import numpy as np
#import matplotlib.pyplot as plt

def SmoothFuncFourier1D(x, n, periodic=False):
    """ Generate smooth function by selecting n random Fourier coefficients.
    The function is assumed
    
    Input
    -----
        x : np.array
            Our spatial interval
        n : int
            desired number of Fourier terms
    Output
    ------
        f : np.array
            values of the generated function on x
    Optional
    --------
        periodic : bool
            if false, then a linear offset is added to break symmetry
    """
    
    # Length of interval
    L = x[len(x)-1] - x[0]

    # Create an array to store our function
    f = np.zeros(len(x))
    
    # Array of Fourier coeffs
    c1 = np.random.randn(n)
    c2 = np.random.randn(n)
    m = np.random.uniform(-10, 10)
    
    for n in range(n):
        f += c1[n] * np.sin(2*n*np.pi*x / L) + c2[n] * np.cos(2*n*np.pi*x / L)
    
    # We add a linear portion to break the symmetry
    if periodic == False:
        f += m * x

    return f

def SmoothFuncFourier2D(X, Y, n):
    """ Generate smooth function by selecting n random Fourier coefficients.
    Input:
        X, Y : np.arrays from np.meshgrid(x, y, indexing='ij')
            Our spatial intervals
        n : int
            desired number of Fourier terms
    Output:
        f : np.array
            values of the generated function on x
    """
    
    # Length of interval
    Lx = X[len(X)-1] - X[0]
    Ly = Y[0,len(X)-1] - Y[0,0]
    
    # Create an array to store our function
    f = np.zeros((len(X), len(Y[0])))
    
    # Array of Fourier coeffs
    c1 = np.random.randn(n)
    c2 = np.random.randn(n)    
    mx = np.random.uniform(-1, 1)
    my = np.random.uniform(-1, 1)
    
    for n in range(n):
        f += (c1[n] * np.sin(n*np.pi*X / Lx) * np.sin(n*np.pi*Y / Ly)
              + c2[n] * np.cos(n*np.pi*X / Lx) * np.cos(n*np.pi*Y / Ly)) 

    # We add a linear portion to break the symmetry
    f += mx * X + my * Y

    return f

def scaled_poly(X, Y, Nx, Ny, norm):
    """this one returns function, not an array"""
    ax = np.random.randn(Nx)
    ay = np.random.randn(Ny)
    
    zx = max(ax / ax[0])
    zy = max(ay / ay[0])
    
    for l in range(Nx):
        ax[l] *= zx**l
    for l in range(Ny):
        ay[l] *= zy**l
        
    f = np.zeros((len(X), len(Y[0])))
    
    for l in range(Nx):
        f += ax[l] * X**l
    for l in range(Ny):
        f += ay[l] * Y**l
        
    return norm*f / abs(f).max()