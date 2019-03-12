# -*- coding: utf-8 -*-

"""
This is an example for the application of the TT and QTT format to Markovian master equations of chemical reaction
networks. For more details, see [1]_.

References
----------
..[1] P. Gelß. "The Tensor-Train Format and Its Applications: Modeling and Analysis of Chemical Reaction
      Networks, Catalytic Processes, Fluid Flows, and Brownian Dynamics", Freie Universität Berlin, 2017
"""

from scikit_tt.tensor_train import TT
import scikit_tt.tensor_train as tt
import scikit_tt.models as mdl
import scikit_tt.solvers.ode as ode
import scikit_tt.utils as utl
import numpy as np
import matplotlib.pyplot as plt


def mean_concentrations(series):
    """Mean concentrations of TT series

    Compute mean concentrations of a given time series in TT format representing probability distributions of, e.g., a
    chemical reaction network..

    Parameters
    ----------
    series: list of instances of TT class

    Returns
    -------
    mean: ndarray(#time_steps,#species)
        mean concentrations of the species over time
    """

    # define array
    mean = np.zeros([len(series), series[0].order])

    # loop over time steps
    for i in range(len(series)):

        # loop over species
        for j in range(series[0].order):
            # define tensor train to compute mean concentration of jth species
            cores = [np.ones([1, series[0].row_dims[k], 1, 1]) for k in range(series[0].order)]
            cores[j] = np.zeros([1, series[0].row_dims[j], 1, 1])
            cores[j][0, :, 0, 0] = np.arange(series[0].row_dims[j])
            tensor_mean = TT(cores)

            # define entry of mean
            mean[i, j] = series[i].transpose() @ tensor_mean

    return mean


utl.header(title='Signaling cascade')

# parameters
# ----------

order = 20
tt_rank = 4
qtt_rank = 12
step_sizes = [1] * 300
qtt_modes = [[2] * 6] * order
threshold = 1e-14

# operator in TT format
# ---------------------

operator = mdl.signaling_cascade(order)

# initial distribution in TT format
# ---------------------------------

initial_distribution = tt.zeros(operator.col_dims, [1] * order)
for p in range(initial_distribution.order):
    initial_distribution.cores[p][0, 0, 0, 0] = 1

# initial guess in TT format
# --------------------------

initial_guess = tt.ones(operator.col_dims, [1] * order, ranks=tt_rank).ortho_right()

# solve Markovian master equation in TT format
# --------------------------------------------

print('TT approach:\n')
solution = ode.trapezoidal_rule(operator, initial_distribution, initial_guess, step_sizes)

# operator in QTT format
# ----------------------

operator = TT.tt2qtt(operator, qtt_modes, qtt_modes, threshold=threshold)

# initial distribution in QTT format
# ----------------------------------

initial_distribution = tt.zeros(operator.col_dims, [1] * operator.order)
for p in range(initial_distribution.order):
    initial_distribution.cores[p][0, 0, 0, 0] = 1

# initial guess in QTT format
# ---------------------------

initial_guess = tt.ones(operator.col_dims, [1] * operator.order, ranks=qtt_rank).ortho_right()

# solve Markovian master equation in QTT format
# ---------------------------------------------

print('\nQTT approach:\n')
solution = ode.trapezoidal_rule(operator, initial_distribution, initial_guess, step_sizes)

# convert to TT and compute mean concentrations
# ---------------------------------------------

for p in range(len(solution)):
    solution[p] = TT.qtt2tt(solution[p], [len(qtt_modes[0])] * order)
mean_concentrations = mean_concentrations(solution)

# plot mean concentrations
# ------------------------

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'axes.grid': True})
plt.plot(np.insert(np.cumsum(step_sizes), 0, 0), mean_concentrations)
plt.title('Mean concentrations', y=1.05)
plt.xlabel(r'$t$')
plt.ylabel(r'$\overline{x_i}(t)$')
plt.axis([0, len(step_sizes), 0, np.amax(mean_concentrations) + 0.5])
plt.legend(['species ' + str(i) for i in range(1, np.amin([8, order]))] + ['...'], loc=4)
plt.show()
