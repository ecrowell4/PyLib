import numpy as np
from numba import jit

@jit(nopython=True)
def laplacian(f:complex, dx:float)->complex:
    """Returns laplacian of complex valued function `y(x)`
    Dirichlet b.c.s are assumed, implying laplacian vanishes
    at the boundary."""
    N:int = len(f)
    ddf:complex = np.zeros(N) + 0j
    ddf[1:-1] = (f[2:] - 2*f[1:-1] + f[:-2]) / dx**2
    return ddf


@jit(nopython=True)
def coulomb_convolve(y:complex, Uc:complex, x:float)->complex:
    """Returns the convolution of two complex valued functions
    y*h.

    Input
        y : np.array
            function to be convolved
        Uc : np.array
            Coulomb kernel to be convoled
        x : np.array
            spatial array

    Output
        g : np.array
            convolution of y with coulomb kernel.
    """
    N:int = len(x)
    res:complex = np.zeros(N) + 0j
    for i in range(N):
        Uc_roll:complex = np.roll(Uc, i)
        Uc_roll[:i] = Uc[1:i+1][::-1]
        res[i] = my_simps(y * Uc_roll[::-1], x)
    return res


@jit(nopython=True)
def subtract_lagrange(f:complex, y:complex, x:float)->complex:
    """Subtacts off the set of lagrange multipliers Y from y.

    Input
        f : np.array
            the variation of some functional F.
            In context of HF theory, f[i] is action of HF operator each
            orbital
        y : np.array
            set of orbitals
        x : np.array
            spatial array

    Output
        F : np.array
            the variation of functional with constraints imposed.
    """

    Norb:int = len(f)
    F:complex = f
    for i in range(Norb):
        for j in range(Norb):
            F[i] -= braket(y[j], f[i], x) * y[j] / braket(y[j], y[j], x)
    return F

@jit(nopython=True)
def gram_schmidt_jit(psi:complex, x:float)->complex:
    """Takes in a set of basis functions and returns an orthonormal basis.

    Input
        x : np.array
            spatial array
        psi : np.array
            array whose elements are the basis functions

    Output
        psi_gm : np.array
            array of orthonormal basis functions
    """
    N:int = len(psi)
    psi_gm:complex = np.zeros(psi.shape) + 0j
    psi_gm[0] = psi[0]
    for k in range(N):
        psi_gm[k] = psi[k]
        for j in range(k):
            psi_gm[k] -= (braket_jit(psi[k], psi_gm[j], x) 
                / braket_jit(psi_gm[j], psi_gm[j], x)) * psi_gm[j]
        psi_gm[k] /= np.sqrt(braket_jit(psi_gm[k], psi_gm[k], x))
    return psi_gm

@jit(nopython=True)
def braket(y:complex, f:complex, x:float)->complex:
    """Computes integral <y|f> = <f|y>*.

    Input
        y,f : np.array
            functions to be integrated.
        x : np.array
            spatial array

    Output
        <y|f> : np.float
            projection of psia onto psib
    """

    return my_simps(y.conjugate()*f, x)

@jit(nopython=True)
def my_simps(f: complex, x:float)->complex:
    """Returns the integral of f over the domain x using the simpsons
    rule. Note that simpsons rule requires an even
    number of intervals. However, we typpically choose and odd number
    (even number of points). Thus, we use simpsons rule for all but the
    last segment and use a trapezoidal on the last segment. Then we do
    simpsons on all but first and use trapz on first. We then average
    the two results.    

    Input
        f : np.array
            function to be integrated
        x : np.aray
            spatial domain
        N : int
            len(x)

    Output
        int(f) : np.float
            integral of f over domain x
    """
    
    N:int = len(x)
    if N%2 == 0:
        dx : float = x[1] - x[0]
        result_left = f[0]
        for j in range(1, N-2, 2):
            result_left = result_left + 4 * f[j]
        for i in range(2, N-3, 2):
            result_left = result_left + 2 * f[i]
        result_left = result_left + f[-2]
        result_left = result_left * dx / 3
        
        result_left = result_left + (f[-2] + f[-1]) * dx / 2
        
        result_right = f[1]
        for j in range(2, N-1, 2):
            result_right = result_right + 4 * f[j]
        for i in range(3, N-2, 2):
            result_right = result_right + 2 * f[i]
        result_right = result_right + f[-1]
        result_right = result_right * dx / 3
        
        result_right = result_right + (f[0] + f[1]) * dx / 2
        return 0.5 * (result_left + result_right)
    else:
        dx : float = x[1] - x[0]
        result = f[0]
        for j in range(1, N, 2):
            result = result + 4 * f[j]
        for i in range(2, N-1, 2):
            result = result + 2 * f[i]
        result = result + f[-1]
        result = result * dx / 3
        return result

def smooth_func(x, n, periodic=False):
    """ Generate smooth function by selecting n random Fourier coefficients.
    The function is assumed
    
    Input
        x : np.array
            Our spatial interval
        n : int
            desired number of Fourier terms
            
    Output
        f : np.array
            values of the generated function on x
            
    Optional
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
    
    for i in range(n):
        f += c1[i] * np.sin(2*i*np.pi*x / L) + c2[i] * np.cos(2*i*np.pi*x / L)
    
    # We add a linear portion to break the symmetry
    if periodic == False:
        f += m * x

    return f