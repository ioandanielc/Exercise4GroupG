import numpy as np
from scipy.optimize import fsolve


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

def I2(mu0, mu1, beta, A, d, nu, b):
    d0 = d + nu + mu0
    d1 = d + nu + mu1
    A_ = d0 * (beta - nu)
    B_ = (d0 - beta) * A + (beta - nu) * d1 * b
    delta0_ = (beta - nu)**2 * d1**2 * b**2 - 2*A*(beta - nu) * ( beta*(mu1-mu0) + d0*(d1-beta))*b + A*A*(beta-d0)**2
    return (-B_ + np.sqrt(delta0_)) / 2 / A_

def I_star(mu0, mu1, beta, A, d, nu, b):
    A_ = (d + nu + mu0) * (beta - nu)
    B_ = (d + nu + mu0 - beta) * A + (beta - nu) * (d + nu + mu1) * b
    return - B_ / 2 / A_

def hopf_bif_func(b, mu0, mu1, beta, A, d, nu):
    """
    Function used to find a Hopf bifurcation on the SIR model.
    This bifurcation happens with these parameters if this function returns 0.

    Args:
        b (scalar): number of beds per 10,000 persons
        ...

    Returns:
        scalar: difference between h's highest zero and I_2
    """
    
    # Check that th 4.4's hypothesis are respected 
    if not (mu1 - mu0 - 2*d)*A / (beta - nu) / d > 0:
        raise ValueError("not valid parameters for th 4.4")
    if not mu1 - mu0 - 4*d > 0:
        raise ValueError("not valid parameters for th 4.4")
    
    # computes the point I_2
    i2 = I2(mu0, mu1, beta, A, d, nu, b)
    # h_ is the function h: I -> h(i)
    h_ = lambda i: h(i, mu0, mu1, beta, A, d, nu, b)
    
    # Get the zeros of h(I), which determine the type of bifurcation in case I2 or I_star are equal to one of the zeros.
    # In the context of the exercice, it is reasonable to give [0.0, 0.05] as initial points
    # in a more general manner, it could be benificial to do a first study of h, knowing it is a 3rd degree polynomial...
    _, hM = fsolve(h_, [0.0, 0.05])
    return hM - i2

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
