import numpy as np
from scipy import interpolate
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
def coulomb_convolve(y:complex, Uc:float, x:float)->complex:
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
        Uc_:complex = np.zeros(N)
        Uc_[:i] = Uc[1:i+1][::-1]
        Uc_[i:] = Uc[:N-i]
        res[i] = my_simps(y * Uc_, x)
    return res

@jit(nopython=True)
def project(f:complex, y:complex, x:float):
    """Returns component of f that is along y.

    Input
        f, y : np.array

    Output
        The component of f along y: <f|y>/<y|y>
    """

    return braket(y, f, x) * y / braket(y, y, x)

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
def overlap(psi:complex, x:float)->complex:
    """ Returns the overlap of all states contained in psi. For orthonormal
    states, this should be equal to the idential operator.

    Input
        psi : np.array
            collection of states
        x : np.array
            spatial array

    Output
        S : np.array
            overlap matrix <n|m>
    """
    Norb:int = len(psi)
    S:complex = np.zeros((Norb, Norb)) + 0j
    for i in range(Norb):
        for j in range(i,Norb):
            S[i,j] = braket(psi[i], psi[j], x)
            S[j,i] = S[i,j].conjugate()
    return S

@jit(nopython=True)
def gram_schmidt(psi:complex, x:float)->complex:
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
            psi_gm[k] -= (braket(psi[k], psi_gm[j], x) 
                / braket(psi_gm[j], psi_gm[j], x)) * psi_gm[j]
        psi_gm[k] /= np.sqrt(braket(psi_gm[k], psi_gm[k], x))
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

def refine_grid(x, f, N_):
    """Takes a an array f that is a function sampled over the domain x,
    and interpolates to a finer grid of N_points.

    Input
        x : np.array
            initial spatial grid
        f : np.array
            function sampled over grid
        N_ : np.array
            desired number of points in refined grid

    Output
        x_ : np.array
            refined spatial array over same domain, but with N_ points sampled
        f_ : np.array
            function interpolated to refined domain.
    """
    assert N_ >= len(x), "You're trying to refine to a coarser grid!"
    f = interpolate.interp1d(x, f, 'cubic')
    L = x[-1] - x[0]
    dx_ = L / (N_ - 1)
    x_ = np.arange(N_) * dx_
    return x_, f(x_) + 0j

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

def diagblock(v, k=0):
    """ Creates a block diagonal matrix, with the elements of v
    as the diagonals.

    Input
        v : np.array
            array of matrices. v[0], v[1], ... are the first, second, ...
            diagonal elements of matrix.
    """
    import numpy as np
    
    shapes = np.array([a.shape for a in v])
    out = np.zeros(np.sum(shapes, axis=0) + abs(k)*shapes[0], dtype=v[0].dtype)

    if k >= 0:
        r, c = 0, abs(k)*shapes[0][0]
    else:
        r, c = abs(k)*shapes[0][0], 0
    for i, (rr, cc) in enumerate(shapes):
        out[r:r + rr, c:c + cc] = v[i]
        r += rr
        c += cc
    return out