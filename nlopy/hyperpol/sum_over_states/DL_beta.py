def betaintDL(x, psi0, E10, nounits = True, trimL = False, trimR = False, 
              trim_man = [0,0], diagnostic = False, getF = False):
    """Calculates the DL function F(x) then computes the nonlinear response
    beta. This code is currently limited to one dimension.
    
    Input:
        x = domain over which psi0 is defined
        psi0 = array representing the ground state wave function. It is 
                important that psi0 be defined on a sufficient, but not 
                overly large domain. Specifically, the end points must be zero
                or very nearly zero.
        E10 = first energy difference used to calculate beta_max
        
    Output:
        betaint 
        
    Optional:
        nounits = True
        trimL = False
        trimR = False
            Enables optimization of the domain for integration of beta. The 
            user must still enter a reasonble domain as a first guess. If the
            wavefunction inputted does not decay exponentially then do not use 
            the trimming algorithm.
        diagnostic = False
            Plots the pertinent integrand so that the user can determine if a 
            good domain was used.
    
    Notes:
        Leave trimL and trimR as False for infinite potential boundary 
        conditions.
    """
    
    import numpy as np
    from scipy.integrate import simps
    from matplotlib import pyplot as plt
    
    hbar = 6.58211928E-16 #eV.s
    m = 0.51099891E6 / (2.99792458E10)**2 #eV.s^2/cm^2
    e = 1E-18 * 4.803204251E-10 #1E-18 statC
    if nounits == True:
        hbar = 1.
        m = 1.
        e = 1.
    
    #First calculate the first order energy shift, x_00
    x00 = simps(x*psi0**2, x)    
    
    #The first integral is has a definite lower bound but an indefinite upper
    #   bound and therefore is represented by an array of integrals each 
    #   beginning at the left side of the domain.
    int1 = np.array([np.trapz( (x[:i]-x00)*psi0[:i]**2, x[:i] ) 
                                for i in np.arange(1, len(x)+1)])
    
    #The next integral to determine the function F(x) is indefinite and is 
    #   allowed to have an undetermined constant so we take the lower bound to
    #   be the left side of the domain without loss of generality. We exclude 
    #   the endpoints anticipating a divergence in the inverse square psi0
    F = 2*m/hbar**2 * np.array(
        [np.trapz( psi0[1:i]**(-2)*int1[1:i], x[1:i])
                                for i in np.arange(2, len(x))]
                              )
    F = np.append(0,np.append(F,0))
    
    #Trim the space to the region where the integrand has converged as much as
    #   as it is able to.
    #Walking in from the left, the integrand must vanish at the proper end
    #   points so we clip until the integrand begins increasing.
    if trimL == True:    
        i = 1    
        integrand = x*F**2*psi0**2
        while True:
            slope = np.polyfit(
                    x[i:i+5], abs(integrand[i:i+5]), 1)[0]
            if slope >= 0:
                cutleft = i+2
                break
            else: 
                i+=1
        x = x[cutleft:]
        F = F[cutleft:]
        psi0 = psi0[cutleft:]
    if trimR == True:
        integrand = x*F**2*psi0**2
        i = -2
        while True:
            slope = np.polyfit(
                    x[i-5:i], abs(integrand[i-5:i]), 1)[0]
            if slope <= 0:
                cutright = i-2
                break
            else:
                i-=1
        x = x[:cutright]
        F = F[:cutright]
        psi0 = psi0[:cutright]
    if trim_man != [0,0]:
        for i in np.arange(len(x)):
            if x[i] >= trim_man[0]:
                cutleft = i
                break
        for j in np.arange(len(x))[::-1]:
            if x[j] <= trim_man[1]:
                cutright = j
                break
        x = x[cutleft:cutright]
        F = F[cutleft:cutright]
        psi0 = psi0[cutleft:cutright]
    
    F = F - simps(F*psi0**2,x)
    
    #Now calculate the total response with the last round of integrals
    betaint = 3*e**3*(simps((F**2 *(x-x00)* psi0**2), x) 
                    ) / (3.**(0.25) * (e*hbar/np.sqrt(m))**3 /E10**(3.5) )
           
    #Plots the pertinent integrand so the user can be assured of convergence.
    if diagnostic == True:
        plt.figure()
        plt.plot(x, psi0**2, label = '$\psi_0^2$')
        plt.plot(x, x*F**2*psi0**2, label = '$xF^2\psi_0^2$')
        plt.legend()           
           
    if getF == True:
        return betaint, F
        
    return betaint
    
    