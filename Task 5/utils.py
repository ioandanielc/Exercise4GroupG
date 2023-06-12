import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sir_model import model
from scipy.integrate import solve_ivp

def solve_scipy(f_ode, y0, time):
    """
    Makes scipy's solve_ivp behave like our solve_euler
    """
    fun = lambda t, y: f_ode(y)
    # Radau method used as it is / can be a stiff pb
    sol = solve_ivp(fun, [time.min(), time.max()], y0, t_eval=time, method="Radau")
    return sol.y, sol.t


def solve_euler(f_ode, y0, time):
    """
    Solves the given ODE system in f_ode using forward Euler.
    :param f_ode: the right hand side of the ordinary differential equation d/dt x = f_ode(x(t)).
    :param y0: the initial condition to start the solution at.
    :param time: np.array of time values (equally spaced), where the solution must be obtained.
    :returns: (solution[time,values], time) tuple.
    """
    yt = np.zeros((len(time), len(y0)))
    yt[0, :] = y0
    step_size = time[1]-time[0]
    for k in range(1, len(time)):
        #print(f_ode(yt[k-1, :]))
        yt[k, :] = yt[k-1, :] + step_size * f_ode(yt[k-1, :])
    return yt, time


def plot_phase_portrait(A, X, Y):
    """
    Plots a linear vector field in a streamplot, defined with X and Y coordinates and the matrix A.
    """
    UV = A@np.row_stack([X.ravel(), Y.ravel()])
    U = UV[0,:].reshape(X.shape)
    V = UV[1,:].reshape(X.shape)

    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])

    #  Varying density along a streamline
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.streamplot(X, Y, U, V, density=[0.5, 1])
    ax0.set_title('Streamplot for linear vector field A*x')
    ax0.set_aspect(1)
    return ax0

def bif_vis(bs3, SIR_0, ts):
    """
    Plot 3 figs of the infected against the susceptibles numbers, with varying 'b'

    Parameters
    -------
    bs3
        a list of 3 values for b
    SIR_0
        initial state of the SIR model
    ts
        np.array of time evenly spaced
    """
    _, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for i in range(3):
        SIR_model = lambda y: model(y, b = bs3[i])
        yt, _ = solve_scipy(SIR_model, SIR_0, ts)
        axs[i].plot(yt[0, :], yt[1, :], c='red', label="SI(time)")
        axs[i].set_title(f"b={bs3[i]:.3f}")
        axs[i].set_xlabel("Susceptibles")
        axs[i].set_ylabel("Infected")
        axs[i].legend()

        """
        X = np.linspace(yt[:, 0].min(), yt[:, 0].max(), 100)
        Y = np.linspace(yt[:, 1].min(), yt[:, 1].max(), 100)
        Z = np.linspace(yt[:, 2].min(), yt[:, 2].max(), 100)

        grid = np.meshgrid(X, Y, Z)

        U, V, _ = SIR_model(np.array(grid))
        axs[i].streamplot(grid[0][:, :, 50], grid[1][:, :, 50], U[:, :, 50], V[:, :, 50], density=[0.5, 0.5])
        """


