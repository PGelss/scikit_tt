# -*- coding: utf-8 -*-

"""
This is an example for the application of the QTT format to Markovian master equations of chemical reaction
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


utl.header(title='Two-step destruction')

# parameters
# ----------

m = 3
step_sizes = [0.001] * 100 + [0.1] * 9 + [1] * 9
qtt_rank = 10
max_rank = 25

# construct operator in TT format and convert to QTT format
# ---------------------------------------------------------

operator = mdl.two_step_destruction(1, 2, 1, m).tt2qtt([[2] * m] + [[2] * (m + 1)] + [[2] * m] + [[2] * m],
                                                       [[2] * m] + [[2] * (m + 1)] + [[2] * m] + [[2] * m],
                                                       threshold=10 ** -14)

# initial distribution in TT format and convert to QTT format
# -----------------------------------------------------------

initial_distribution = tt.zeros([2 ** m, 2 ** (m + 1), 2 ** m, 2 ** m], [1] * 4)
initial_distribution.cores[0][0, -1, 0, 0] = 1
initial_distribution.cores[1][0, -2, 0, 0] = 1
initial_distribution.cores[2][0, 0, 0, 0] = 1
initial_distribution.cores[3][0, 0, 0, 0] = 1
initial_distribution = TT.tt2qtt(initial_distribution, [[2] * m] + [[2] * (m + 1)] + [[2] * m] + [[2] * m],
                                 [[1] * m] + [[1] * (m + 1)] + [[1] * m] + [[1] * m], threshold=0)

# initial guess in QTT format
# ---------------------------

initial_guess = tt.uniform([2] * (4 * m + 1), ranks=qtt_rank).ortho_right()

# solve Markovian master equation in QTT format
# ---------------------------------------------


solution = ode.implicit_euler(operator, initial_distribution, initial_guess, step_sizes, tt_solver='mals',
                              threshold=1e-10, max_rank=max_rank)

# compute approximation errors
errors = ode.errors_impl_euler(operator, solution, step_sizes)
print('Maximum error: ' + str("%.2e" % np.amax(errors)) + '\n')

# convert to TT and compute mean concentrations
# ---------------------------------------------

for p in range(len(solution)):
    solution[p] = TT.qtt2tt(solution[p], [m, m + 1, m, m])
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
plt.axis([0, 2, 0, 2 ** (m + 1) - 2])
plt.legend(['species ' + str(i) for i in range(1, 5)], loc=1)
plt.show()
