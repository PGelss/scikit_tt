# -*- coding: utf-8 -*-

"""
This is an example of the application of MANDy to a high-dimensional dynamical system. See [1]_ for details.

References
----------
.. [1] P. Gelß, S. Klus, J. Eisert, C. Schütte, "Multidimensional Approximation of Nonlinear Dynamical Systems",
       arXiv:1809.02448, 2018
"""

from __future__ import division
import numpy as np
import scipy.linalg as splin
import scikit_tt.data_driven.regression as reg
import scikit_tt.models as mdl
import scikit_tt.utils as utl
import matplotlib.pyplot as plt
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


utl.header(title='MANDy - Fermi-Pasta-Ulam problem', subtitle='Example 1')

# model parameters
number_of_oscillators = 10
psi = [lambda t: 1, lambda t: t, lambda t: t ** 2, lambda t: t ** 3]
p = len(psi)

# construct exact solution in TT and matrix format
start_time = utl.progress('Construct exact solution in TT format', 0)
xi_exact = mdl.fpu_coefficients(number_of_oscillators)
utl.progress('Construct exact solution in TT format', 100, cpu_time=_time.time() - start_time)
start_time = utl.progress('Construct exact solution in matrix format', 0)
xi_exact_mat = xi_exact.full().reshape([p ** number_of_oscillators, number_of_oscillators])
utl.progress('Construct exact solution in matrix format', 100, cpu_time=_time.time() - start_time)

# number of repeats
repeats = 1

# snapshot parameters
snapshots_min = 1000
snapshots_max = 6000
snapshots_step = 500

# maximum number of snapshots for matrix approach
snapshots_mat = 5000

# define arrays for CPU times and relative errors
cpu_times = np.zeros([6, int((snapshots_max - snapshots_min) / snapshots_step) + 1])
rel_errors = np.zeros([6, int((snapshots_max - snapshots_min) / snapshots_step) + 1])

# find maximum index for matrix computations (used for plotting)
if snapshots_max > snapshots_mat:
    index_mat = int(np.ceil((snapshots_mat - snapshots_min) / snapshots_step + 1e-10))
else:
    index_mat = int((snapshots_max - snapshots_min) / snapshots_step) + 1

# compare CPU times of tensor-based and matrix-based approaches
for i in range(snapshots_min, snapshots_max + snapshots_step, snapshots_step):

    for r in range(repeats):

        print('\nNumber of snapshots: ' + str(i))
        print('-' * (21 + len(str(i))) + '\n')

        # storing index
        ind = int((i - snapshots_min) / snapshots_step)

        # generate data
        start_time = utl.progress('Generate test data', 0)
        [x, y] = fermi_pasta_ulam(number_of_oscillators, i)
        utl.progress('Generate test data', 100, cpu_time=_time.time() - start_time)

        # computation in matrix format
        if i <= snapshots_mat:
            start_time = utl.progress('Running matrix-based MANDy', 0)

            # construct psi_x in matrix format
            psi_x = np.zeros([p ** number_of_oscillators, i])
            for j in range(i):
                c = [psi[l](x[0, j]) for l in range(p)]
                for k in range(1, number_of_oscillators):
                    c = np.tensordot(c, [psi[l](x[k, j]) for l in range(p)], axes=0)
                psi_x[:, j] = c.reshape(psi_x.shape[0])

            utl.progress('Running matrix-based MANDy', 50, cpu_time=_time.time() - start_time)

            # compute xi in matrix format
            with utl.timer() as time:
                [u, s, v] = splin.svd(psi_x, full_matrices=False, overwrite_a=True, check_finite=False,
                                      lapack_driver='gesdd')
                del psi_x
                xi = y @ v.transpose() @ np.diag(np.reciprocal(s)) @ u.transpose()
            cpu_times[0, ind] += time.elapsed / repeats
            if r == 0:
                rel_errors[0, ind] = np.linalg.norm(xi.transpose() - xi_exact_mat) / np.linalg.norm(xi_exact_mat)
            del xi, u, s, v
            utl.progress('Running matrix-based MANDy', 100, cpu_time=_time.time() - start_time)
            print('   CPU time      : ' + str("%.2f" % time.elapsed) + 's')
            print('   relative error: ' + str("%.2e" % rel_errors[0, ind]))

        else:

            # extrapolate cpu times of matrix approach
            cpu_times[0, ind] = 2 * cpu_times[0, ind - 1] - cpu_times[0, ind - 2]

            # exact computation in TT format
            start_time = utl.progress('Running MANDy (eps=0)', 0)
        with utl.timer() as time:
            xi = reg.mandy_cm(x, y, psi, threshold=0)
        cpu_times[1, ind] += time.elapsed / repeats
        if r == 0:
            rel_errors[1, ind] = (xi - xi_exact).norm() / xi_exact.norm()
        del xi
        utl.progress('Running MANDy (eps=0)', 100, cpu_time=_time.time() - start_time)
        print('   CPU time      : ' + str("%.2f" % time.elapsed) + 's')
        print('   relative error: ' + str("%.2e" % rel_errors[1, ind]))

        # use thresholds larger 0 for orthonormalizations
        for j in range(0, 4):
            start_time = utl.progress('Running MANDy (eps=10^' + str(-10 + j) + ')', 0)
            with utl.timer() as time:
                xi = reg.mandy_cm(x, y, psi, threshold=10 ** (-10 + j))
            cpu_times[j + 2, ind] += time.elapsed / repeats
            if r == 0:
                rel_errors[j + 2, ind] = (xi - xi_exact).norm() / xi_exact.norm()
            del xi
            utl.progress('Running MANDy (eps=10^' + str(-10 + j) + ')', 100, cpu_time=_time.time() - start_time)
            print('   CPU time      : ' + str("%.2f" % time.elapsed) + 's')
            print('   relative error: ' + str("%.2e" % rel_errors[j + 2, ind]))

# plot results
# ------------

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'axes.grid': True})

# CPU times
f = plt.figure(dpi=300)
x_values = np.arange(snapshots_min, snapshots_max + 1, snapshots_step)
plt.plot(x_values[:index_mat], cpu_times[0, :index_mat], 'o-', label=r"Matrix repr.")
plt.plot(x_values[index_mat - 1:], cpu_times[0, index_mat - 1:], 'C0--')
plt.plot(x_values, cpu_times[1, :], 'o-', label=r"TT - exact")
plt.plot(x_values, cpu_times[2, :], 'o-', label=r"TT - $\varepsilon=10^{-10}$")
plt.plot(x_values, cpu_times[3, :], 'o-', label=r"TT - $\varepsilon=10^{-9}$")
plt.plot(x_values, cpu_times[4, :], 'o-', label=r"TT - $\varepsilon=10^{-8}$")
plt.plot(x_values, cpu_times[5, :], 'o-', label=r"TT - $\varepsilon=10^{-7}$")
plt.gca().set_ylim(bottom=0)
plt.title(r'CPU times for $d = 10$', y=1.03, fontsize=22)
plt.xlabel(r'$m$', fontsize=20)
plt.ylabel(r'$T / s$', fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.axis([snapshots_min, snapshots_max, 0, 2000])
plt.legend(loc=2, fontsize=12).get_frame().set_alpha(1)
plt.show()

# relative errors
plt.figure(dpi=300)
x_values = np.arange(snapshots_min, snapshots_max + 1, snapshots_step)
plt.semilogy(x_values[:index_mat], rel_errors[0, :index_mat], 'o-', label="Matrix - exact")
for j in range(5):
    exp = -11 + j
    if exp == -11:
        plt.semilogy(x_values[:index_mat], rel_errors[1, :index_mat], 'o-', label=r'TT - exact')
    else:
        plt.semilogy(x_values[:index_mat], rel_errors[1 + j, :index_mat], 'o-',
                     label=r'TT - $\varepsilon = 10^{' + str(exp) + '}$')
plt.grid(which='minor')
plt.title(r'Relative errors for $d = 10$', y=1.03, fontsize=22)
plt.xlabel(r'$m$', fontsize=20)
plt.ylabel(r'$|| \Xi_{\textrm{exact}} - \Xi ||~/~|| \Xi_{\textrm{exact}} ||$', fontsize=20)
plt.legend(loc=1, fontsize=12).get_frame().set_alpha(1)
plt.axis([snapshots_min, snapshots_mat, 1e-4, 1e-1])
plt.xticks([1000, 2000, 3000, 4000, 5000], fontsize=20)
plt.yticks(fontsize=20)
plt.show()
