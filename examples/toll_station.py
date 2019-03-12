# -*- coding: utf-8 -*-

"""
This is an example for the application of the TT format to a nearest-neighbor interaction system. For more details, see
[1]_.

References
----------
.. [1] P. Gelß, S. Klus, S. Matera, C. Schütte, "Nearest-neighbor interaction systems in the tensor-train format",
       Journal of Computational Physics, 2017
"""

from scikit_tt.tensor_train import TT
import scikit_tt.tensor_train as tt
import scikit_tt.models as mdl
import scikit_tt.solvers.ode as ode
import scikit_tt.utils as utl
import numpy as np
import matplotlib.pyplot as plt
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D


def average_numbers_of_cars(series):
    # define array
    average_noc = np.zeros([len(series), series[0].order])

    # loop over time steps
    for i in range(len(series)):

        # loop over species
        for j in range(series[0].order):
            # define tensor train to compute average number of cars
            cores = [np.ones([1, series[0].row_dims[k], 1, 1]) for k in range(series[0].order)]
            cores[j] = np.zeros([1, series[0].row_dims[j], 1, 1])
            cores[j][0, :, 0, 0] = np.arange(series[0].row_dims[j])
            tensor_mean = TT(cores)

            # define entry of average_noc
            average_noc[i, j] = series[i].transpose() @ tensor_mean

    return average_noc


utl.header(title='Toll station')

# parameters
# ----------

number_of_lanes = 20
maximum_number_of_cars = 9
initial_number_of_cars = 5
integration_time = 30

# construct operator
# ---------------------

operator = mdl.toll_station(number_of_lanes, maximum_number_of_cars)

# construct initial distribution
# ------------------------------

initial_distribution = tt.zeros(operator.col_dims, [1] * operator.order)
for p in range(initial_distribution.order):
    initial_distribution.cores[p][0, initial_number_of_cars, 0, 0] = 1

# construct initial guess
# -----------------------

initial_guess = tt.ones(operator.col_dims, [1] * operator.order, ranks=10).ortho_right()

# solve Markovian master equation
# -------------------------------

solution, time_steps = ode.adaptive_step_size(operator, initial_distribution, initial_guess, integration_time,
                                              step_size_first=1e-1, closeness_min=1e-10, error_tol=1e-1)

# plot average numbers of cars per lane and time
# ----------------------------------------------

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'axes.grid': True})
fig = plt.figure()
ax = fig.gca(projection='3d')
X, Y = np.meshgrid(np.arange(1, number_of_lanes + 1), time_steps)
ax.plot_surface(X, Y, average_numbers_of_cars(solution))
ax.view_init(30, 120)
plt.title('Average numbers of cars', y=1.05)
ax.set_xlabel('Lanes')
ax.set_ylabel('Time')
ax.set_zlabel('Cars')
ax.set_xlim(0, number_of_lanes)
ax.set_ylim(0, integration_time)
ax.set_zlim(0, initial_number_of_cars + 0.25)
ax.set_xticks(np.arange(0, number_of_lanes + 5, 5))
ax.set_yticks(np.arange(0, integration_time + 10, 10))
ax.set_zticks(np.arange(0, initial_number_of_cars + 1))
plt.show()
