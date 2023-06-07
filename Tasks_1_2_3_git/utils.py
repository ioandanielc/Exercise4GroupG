import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

############################################################Given Functions#############################################################################
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
    ax0.set_title('Streamplot for linear vector field A*x');
    ax0.set_aspect(1)
    return ax0


############################################################Functions for task 1#############################################################################

#Calulate important quantities for the matrix A (which depends on alpha such as determinatn, eigenvalues and number of eigenvalues with a real part
def present_eigval(alpha, A, plt):
    #Print the determinant
    print("Determinant is: " + str(np.linalg.det(A)))
    #Print the trace
    print("Trace is: " + str(np.trace(A)))
    mes1 = str(round(np.real(np.linalg.eigvals(A))[0], 2)) + " + " + str(round((np.imag(np.linalg.eigvals(A))[0]))) + "j"
    mes2 = str(round(np.real(np.linalg.eigvals(A))[1], 2)) + " + " + str(round((np.imag(np.linalg.eigvals(A))[1]))) + "j"

    #Print the 2 eigenvalues
    print ("Eigenvalue 1 is: " + mes1)
    print ("Eigenvalue 2 is: " + mes2)

    #Count the number of eigenvalues with a positive real part (and analogously negative real part)
    n_plus = np.sum(np.real(np.linalg.eigvals(A)) > 0, axis=0)
    n_minus = np.sum(np.real(np.linalg.eigvals(A)) < 0, axis=0)
    print ("n_+: " + str(n_plus))
    print ("n_-: " + str(n_minus))

    #Plot the eigenvalues graph
    fig, ax = plt.subplots()
    ax.scatter([np.real(np.linalg.eigvals(A))[0], np.real(np.linalg.eigvals(A))[1]], [np.imag(np.linalg.eigvals(A))[0], np.imag(np.linalg.eigvals(A))[1]])
    ax.grid(True, which='both')

    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')

    plt.xlabel('Re(lambda)')
    plt.ylabel('Im(lambda)')

    plt.show()

    #Return the title for the figures
    return ("alpha = " + str(alpha) + ", ev1 = " + str(mes1) + ", ev2 = " + str(mes2))

#Plot the phase portraits
def present_phase_portr(title, A):
    # define notebook parameters
    w = 10
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]

    # example for Euler's method to construct and plot a trajectory over the stream plot
    y0 = np.array([.1, .1])
    y1 = np.array([w + 1, w + 1])

    time = np.linspace(0, 10, 1000)
    yt0, time0 = solve_euler(lambda y: A@y, y0, time)
    yt1, time1 = solve_euler(lambda y: A@y, y1, time)


    # example linear vector field A*x
    ax0 = plot_phase_portrait(A, X, Y)

    # then plot the trajectory over it
    ax0.plot(yt0[:, 0], yt0[:, 1], c='red', label='Trajectory for [.1, .1]')
    ax0.plot(yt1[:, 0], yt1[:, 1], c='orange', label='Trajectory for the upper right corner point')


    #Limit the plot visualization to the calculated area
    ax0.set_xlim([-w, w])
    ax0.set_ylim([-w, w])


    # prettify
    ax0.legend()
    ax0.set_aspect(1)

    plt.xlabel('x_i')
    plt.ylabel('x_j')

    plt.title(title)
    plt.show()


############################################################Functions for task 2#############################################################################

#Plot the bifurcations no matter the chosen speed
def plot_bifurcation_1(alpha, no):
    if no == 1:
        new_alpha = alpha[alpha >= 0]
        new_alpha = np.insert(new_alpha, 0, 0, axis=0)

    elif no == 2:
        new_alpha = alpha[alpha >= 3]
        new_alpha = np.insert(new_alpha, 0, 3, axis=0)

    # Label the axis
    plt.xlabel('alpha')
    plt.ylabel('x')

    # Plot the 2 bifurcations corresponding to steady states
    if no == 1:
        plt.plot(new_alpha, np.sqrt(new_alpha), 'b')
        plt.plot(new_alpha, -np.sqrt(new_alpha), 'b')
    elif no == 2:
        plt.plot(new_alpha, np.sqrt((new_alpha - 3) / 2), 'b')
        plt.plot(new_alpha, -np.sqrt((new_alpha - 3) / 2), 'b')

    #Plot a relevant steady state of x for all the 3 cases

    if no == 1:
        plt.plot([new_alpha[0], new_alpha[-10], new_alpha[-10]], [np.sqrt(new_alpha[0]), np.sqrt(new_alpha[-10]), -np.sqrt(new_alpha[-10])], 'ro')
    elif no == 2:
        plt.plot([new_alpha[0], new_alpha[-10], new_alpha[-10]], [np.sqrt((new_alpha[0] - 3) / 2), np.sqrt((new_alpha[-10] - 3) / 2), -np.sqrt((new_alpha[-10] - 3) / 2)], 'ro')

    ax = plt.gca()
    ax.set_xlim([-2, 12])
    ax.set_ylim([-4, 4])

############################################################Functions for task 3#############################################################################

#A function that plots the Andronov Hopf bifurcation, given an alpha
#the meshgrid and w, which represents the boundaries of the area to be
#streamplotted over
def andronov_hopf(alpha, X, Y, w, density):
    #We calculate the tangent values for the 2 coordinates
    A = [[alpha, -1], [1, alpha]]
    UV = A@np.row_stack([X.ravel(), Y.ravel()])
    U = UV[0,:].reshape(X.shape) + X * (np.square(X) + np.square(Y))
    V = UV[1,:].reshape(X.shape) + Y * (np.square(X) + np.square(Y))


    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 1])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.streamplot(X, Y, U, V, density=[density, density])
    ax0.set_title('Andronov Hopf for alpha = ' + str(alpha));
    ax0.set_aspect(1)

    #We calculate the orbits of the 2 required points from the exercise sheet
    y0 = np.array([2, 0])
    y1 = np.array([0.5, 0])

    time = np.linspace(0, 10, 1000)
    yt0, time0 = solve_euler(lambda y: A@y, y0, time)
    yt1, time1 = solve_euler(lambda y: A@y, y1, time)

    ax0.plot(yt0[:, 0], yt0[:, 1], c='red', label='Trajectory of [2, 0]')
    ax0.plot(yt1[:, 0], yt1[:, 1], c='orange', label='Trajectory of [0.5, 0]')

    ax0.set_xlim([-w, w])
    ax0.set_ylim([-w, w])

    ax0.legend()

    plt.xlabel('x_1')
    plt.ylabel('x_2')

    plt.show()

    return ax0

#A function that plots the cusp bifurcation
def cusp(w, count):
    #The boundaries of the plotted area = w
    #The number of points per coordinate (for my system there is a good speed-visualization
    #trade-off between 20 and 40) = count

    #Getting an uniform represenation of values over x-s and alpha2-s
    xs = np.linspace(-w, w, count)
    alpha2s = np.linspace(-w, w, count)

    #Returns the value of alpha1
    def fun(x, alpha2):
        return x**3 - alpha2 * x


    #A function that colors a data point according to its relative position on the 3 axis
    #wrt to the min and max values on those axes
    def setcolor(x, a2, a1, maxa1, mina1):
        return [((a1-mina1) / (maxa1 - mina1), (w + x)/w/2, (w + a2)/w/2, 1)]

    #We calculate all teh values for alpha1
    alpha1s = []
    points = []
    for x in xs:
        for alpha2 in alpha2s:
            alpha1 = fun(x, alpha2)
            alpha1s.append(alpha1)
            point = (alpha1, alpha2, x)
            points.append(point)

    #We store the min and max values of alpha1, we need this for interpolating the color
    #on the alpha1 axis
    maxa1 = np.max(alpha1s)
    mina1 = np.min(alpha1s)

    #Plot the figure for different perspectives (inefficient, but we have not found another way)
    azims = [0, 45, 90, 135, 180, 225, 270, 315]
    for azim in azims:
        # Creates a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # We plot every point
        for point in points:
            ax.scatter(point[0], point[1], point[2], c=setcolor(point[2], point[1], point[0], maxa1, mina1))

        # Set the labels
        ax.set_zlabel('x')
        ax.set_xlabel('a1')
        ax.set_ylabel('a2')

        # Set the boundaries
        ax.set_xlim([mina1, maxa1])
        ax.set_ylim([-w, w])
        ax.set_zlim([-w, w])

        # We modify the following values (most importantly the azimuth) in order to be able to
        # see the cusp bifurcation better.
        ax.view_init(elev=30, azim=azim, roll=0)

        ax.set_title('Cusp bifurcation (azimuth = ' + str(azim) + ')');

        plt.show()


