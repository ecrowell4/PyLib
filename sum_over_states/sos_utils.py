import numpy as np

def D1(E, omega):
    """Returns the propogator for a first order process.
    
    Input
    	E : np.array
        	array of eigenergies.
        	Note: if omega is take as negative, then user 
        	should pass E.conjugate() instead of E.
    omega : float
        input frequency

    Output
    	D1 : np.array
    		propogator for first order process
    """
    
    # Take E -> E - E0
    E = E - E[0]
    
    # Compute and return the propogator
    return 1 / (E[1:] - omega)

def D2(E, omega1, omega2):
    """Returns propogator for a second order process.
    
    Input
    	E : np.array
        	array of eigenergies.
        	Note: if omega is take as negative, then user 
        	should pass E.conjugate() instead of E.
    omega1, omega2 : float
        input frequencies

    Output
    	D2 : np.array
    		propogator for second order process
    """
    E = E - E[0]
    return 1 / (E[1:] - omega1 - omega2)