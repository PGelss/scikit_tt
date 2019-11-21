# -*- coding: utf-8 -*-

"""
This is an example of the application of MANDy to a high-dimensional dynamical system. See [1]_ for details.

References
----------
.. [1] P. Gelß, S. Klus, J. Eisert, C. Schütte, "Multidimensional Approximation of Nonlinear Dynamical Systems",
       arXiv:1809.02448, 2018
"""

import numpy as np
import scikit_tt.data_driven.regression as reg
import scikit_tt.models as mdl
import scikit_tt.utils as utl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import time as _time


def fermi_pasta_ulam(number_of_oscillators, number_of_snapshots):
    """Fermi–Pasta–Ulam problem.

    Generate data for the Fermi–Pasta–Ulam problem represented by the differential equation

        d^2/dt^2 x_i = (x_i+1 - 2x_i + x_i-1) + 0.7((x_i+1 - x_i)^3 - (x_i-x_i-1)^3).

    See [1]_ for details.

    Parameters
    ----------
    number_of_oscillators: int
        number of oscillators
    number_of_snapshots: int
        number of snapshots

    Returns
    -------
    snapshots: ndarray(number_of_oscillators, number_of_snapshots)
        snapshot matrix containing random displacements of the oscillators in [-0.1,0.1]
    derivatives: ndarray(number_of_oscillators, number_of_snapshots)
        matrix containing the corresponding derivatives

    References
    ----------
    .. [1] P. Gelß, S. Klus, J. Eisert, C. Schütte, "Multidimensional Approximation of Nonlinear Dynamical Systems",
           arXiv:1809.02448, 2018
    """

    # define random snapshot matrix
    snapshots = 0.2 * np.random.rand(number_of_oscillators, number_of_snapshots) - 0.1

    # compute derivatives
    derivatives = np.zeros((number_of_oscillators, number_of_snapshots))
    for j in range(number_of_snapshots):
        derivatives[0, j] = snapshots[1, j] - 2 * snapshots[0, j] + 0.7 * (
                (snapshots[1, j] - snapshots[0, j]) ** 3 - snapshots[0, j] ** 3)
        for i in range(1, number_of_oscillators - 1):
            derivatives[i, j] = snapshots[i + 1, j] - 2 * snapshots[i, j] + snapshots[i - 1, j] + 0.7 * (
                    (snapshots[i + 1, j] - snapshots[i, j]) ** 3 - (snapshots[i, j] - snapshots[i - 1, j]) ** 3)
        derivatives[-1, j] = - 2 * snapshots[-1, j] + snapshots[-2, j] + 0.7 * (
                -snapshots[-1, j] ** 3 - (snapshots[-1, j] - snapshots[-2, j]) ** 3)

    return snapshots, derivatives


utl.header(title='MANDy - Fermi-Pasta-Ulam problem', subtitle='Example 2')

# model parameters
psi = [lambda t: 1, lambda t: t, lambda t: t ** 2, lambda t: t ** 3]
p = len(psi)

# snapshot parameters
snapshots_min = 500
snapshots_max = 6000
snapshots_step = 500

# dimension parameters
d_min = 3
d_max = 20

# define arrays for CPU times and relative errors
rel_errors = np.zeros([int((snapshots_max - snapshots_min) / snapshots_step) + 1, int(d_max - d_min) + 1])

# compare CPU times of tensor-based and matrix-based approaches
for i in range(d_min, d_max + 1):

    print('Number of osciallators: ' + str(i))
    print('-' * (24 + len(str(i))) + '\n')

    # construct exact solution in TT and matrix format
    start_time = utl.progress('Construct exact solution in TT format', 0)
    xi_exact = mdl.fpu_coefficients(i)
    utl.progress('Construct exact solution in TT format', 100, cpu_time=_time.time() - start_time)

    # generate data
    start_time = utl.progress('Generate test data', 0)
    [x, y] = fermi_pasta_ulam(i, snapshots_max)
    utl.progress('Generate test data', 100, cpu_time=_time.time() - start_time)

    start_time = utl.progress('Running MANDy', 0)
    for j in range(snapshots_min, snapshots_max + snapshots_step, snapshots_step):
        # storing indices
        ind_1 = rel_errors.shape[0] - 1 - int((j - snapshots_min) / snapshots_step)
        ind_2 = int((i - d_min))

        utl.progress('Running MANDy', 100 * (rel_errors.shape[0] - ind_1 - 1) / rel_errors.shape[0],
                     cpu_time=_time.time() - start_time)

        # approximate coefficient tensor
        xi = reg.mandy_cm(x[:, :j], y[:, :j], psi, threshold=1e-10)
        rel_errors[ind_1, ind_2] = (xi - xi_exact).norm() / xi_exact.norm()
        del xi

    utl.progress('Running MANDy', 100, cpu_time=_time.time() - start_time)
    print(' ')

# plot results
# ------------

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'axes.grid': True})
plt.figure(dpi=300)
plt.imshow(rel_errors, cmap='jet', norm=LogNorm())
plt.colorbar(aspect=12)
plt.gca().set_aspect('auto')
plt.title(r'Relative errors for $\varepsilon = 10^{-10}$', y=1.03)
plt.grid(False)
plt.xlabel(r'$d$')
plt.ylabel(r'$m$')
plt.xticks(np.arange(1, rel_errors.shape[1], 2), np.arange(d_min + 1, d_max + 1, 2))
plt.yticks(np.arange(rel_errors.shape[0] - 2, -1, -2),
           np.arange(snapshots_min + snapshots_step, snapshots_max + 1, 2 * snapshots_step))
plt.show()
