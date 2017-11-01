"""The following functions identify a variational solution to the Schrodinger
equation in 1D within a definite parameter space.
"""

def Evar(x, psi, V):
    """Calculates the expectation value of the energy for a given state vector
    psi0, in a potential V, on the domain x. Assumes atomic units and a 1D
    kinetic energy.
    """
    import numpy as np
    from scipy.integrate import simps
    
    #Form the matrix representation of the potential in the x basis
    V = np.diag(V[1:-1])
    
    #The vector h represents the grid spacing
    x = np.array(x)   #just in case x isn't a numpy array already
    h = x[1:]-x[:-1]  #shift x right, shift x left, take a difference
    
    #The kinetic energy matrix tailored by an arbitrary, ordered, domain array
    T = ( np.diag(-2 / (h[1:]*h[:-1])) +
        np.diag(1 / (h[1:-1]*(h[1:-1]/2.+h[:-2]/2)),k=1) +
        np.diag(1 / (h[1:-1]*(h[2:]/2.+h[1:-1]/2.)),k=-1))

    H = -1./2. * T + V
    
    return simps(psi[1:-1]*np.dot(H,psi[1:-1]), x[1:-1])/simps(psi**2,x)
    
def variation(x, V, varsol, params, show = False):
    """Calculates the parameters which minimize the energy for a variational 
    ground state function in 1D.  
    
    Input:
        x : (array) Indicates the spatial extent of the domain and imposses 
            Dirichlet boundary conditions at its endpoints
        V : (array) Underlying potential which will determine the solution
        varsol : (function) Function of variational parameters which will be 
            used to constrain the solutions. 
            varsol(x, [a,b,c,...]) where [a,b,c,...] are particular choices
                from params
        params : (array) A list of tuples which represent a paricular choice of
            parameters for the variational solution. An example:
            a = np.arange(0, 10, 0.1)
            b = np.arange(0, 10, 0.1)
            c = np.arange(0, 10, 0.1)
            params = [[i, j, k] for i in a for j in b for k in c]
            
    Output:
        psi : (array) The wave function which minimizes the energy on the 
            domain x for the parameters dictated in params
        E : (float) The resulting energy
        opt_params : (tuple) List of the parameter values which minimize the 
            energy
        
    Optional:
        show = False : Plots x vs psi, the variational solution and prints the 
            energy and optimal parameters
    
    Notes:
    
    """
    import numpy as np
    from matplotlib import pyplot as plt
    from scipy.integrate import simps
    
    #Check that the array x and V are compatible
    if len(x) != len(V):
        print('Error: x and V are incompatible arrays.')
        return 0            
    
    #Calculate all of the energies for the input parameter space. 
    energies = [Evar(x, varsol(x, p)/np.sqrt(simps(varsol(x, p)**2, x)), V) 
                    for p in params]
                        
    #Determine the minimum energy
    arg_opt_params = np.argmin(energies)
    
    varsol_opt = (varsol(x, params[arg_opt_params]) /
                    np.sqrt(simps(varsol(x, params[arg_opt_params])**2, x)) )
    
    if show == True:
        print('Energy minimum = '+str(energies[arg_opt_params]))
        print('Optimum parameters = '+str(params[arg_opt_params]))
        fig = plt.figure()
        plt.plot(x, varsol_opt, label = 'var sol')
        plt.legend()
        plt.show()
    
    #Warning if the minimum energy solution is on the edge of the allowed 
    #   parameter space, indicating the a local minimum was never achieved.
    #NEED    
    
    return varsol_opt, energies[arg_opt_params], params[arg_opt_params]
    
###############################################################################
#EXAMPLE: can run this function or run the code it contains, as you'd like.
def variation_example():
    import numpy as np
    from matplotlib import pyplot as plt
    from NLOpy import SEsolve

    #Define the clipped harmonic oscillator
    x = np.arange(0, 10, 0.1)
    def V(x):
        return x**2
    
    psi, E, xx = SEsolve(x, V(x))
    
    #Define the variational function where p is the set of parameters to vary
    def varsol(x, p):
        return x*np.exp(-p[1]*(x-p[0])**2)
            
    #Define parameter space over which we'd like to look for a minimum energy
    #   solution
    params = [[a,b] for a in np.arange(0,1,0.1) for b in np.arange(0.5,2,0.1)]
    
    #Solve
    varsol_opt, Evar, params_opt = variation(x, V(x), varsol, params, show = True)
    
    #Plot ideal solution for comparison
    plt.plot(x, -psi[0], lw = 2, ls = '--', label = r'$\psi_0$')
    plt.text(4, 0.5, 'Evar = '+str(Evar)+'\n E = '+str(E[0]), fontsize = 10)
    plt.legend()
    plt.show()
    
    return