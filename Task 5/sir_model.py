import numpy as np


def mu(b, I, mu0, mu1):
    """ Computes the recovery rate
    Interpolates between mu0 and mu1 as I changes
    
    Parameters
    -------
    b
        number of beds per 10.000 individuals
    I
        number of infected individuals
    mu0
        mu's minimum (I->inf)
    mu1
        mu's maximum (I=0)

    Returns
    -------
    mu
        infection rate
    """

    return mu0 + (mu1 - mu0) * (b/(I+b))

def R0(beta, d, nu, mu1):
    """
    Basic reproduction number.
    """
    return beta / (d + nu + mu1)

def h(I, mu0, mu1, beta, A, d, nu, b):
    """
    Indicator function for bifurcations.
    """
    c0 = b**2 * d * A
    c1 = b * ((mu0-mu1+2*d) * A + (beta-nu)*b*d)
    c2 = (mu1-mu0)*b*nu + 2*b*d*(beta-nu)+d*A
    c3 = d*(beta-nu)
    res = c0 + c1 * I + c2 * I**2 + c3 * I**3
    return res
    

def model(y, mu0 = 10.0, mu1=10.45, beta=11.5, A=20, d=0.1, nu=1.0, b=0.022):
    """
    SIR model including hospitalization and natural death.
    
    Parameters:
    -----------
    mu0
        Minimum recovery rate
    mu1
        Maximum recovery rate
    beta
        average number of adequate contacts per unit time with infectious individuals
    A
        recruitment rate of susceptibles (e.g. birth rate)
    d
        natural death rate
    nu
        disease induced death rate
    b
        hospital beds per 10,000 persons
    """

    S,I,R = y[:]
    m = mu(b, I, mu0, mu1)

    # total number of individuals
    _N = S + R + I

    # number of susceptible individuals infected in dt
    _d_StoI = beta*S*I/_N 

    # number of infected individuals recovered in dt
    _d_ItoR = m*I
    
    dSdt =     - d*S           - _d_StoI + A
    dIdt = -(d+nu)*I - _d_ItoR + _d_StoI
    dRdt =     - d*R + _d_ItoR 
    
    return np.array([dSdt, dIdt, dRdt])
