# -*- coding: utf-8 -*-

"""
This is an example of the application of MANDy to a high-dimensional dynamical system. See [1]_ for details.

References
----------
.. [1] P. Gelß, S. Klus, J. Eisert, C. Schütte, "Multidimensional Approximation of Nonlinear Dynamical Systems",
       arXiv:1809.02448, 2018
"""

import numpy as np
import scipy.linalg as splin
import scikit_tt.mandy as mandy
import scikit_tt.models as mdl
import scikit_tt.utils as utl
import matplotlib.pyplot as plt

utl.header(title='MANDy - Fermi-Pasta-Ulam problem', subtitle='Example 1')

# model parameters
d = 10
psi = [lambda x: 1, lambda x: x, lambda x: x ** 2, lambda x: x ** 3]
p = len(psi)

# construct exact solution in TT and matrix format
utl.progress('Construct exact solution in TT format', 0, dots=7)
xi_exact = mdl.fermi_pasta_ulam_coefficient_tensor(d)
utl.progress('Construct exact solution in TT format', 100, dots=7)
utl.progress('Construct exact solution in matrix format', 0)
xi_exact_mat = xi_exact.full().reshape([p ** d, d])
utl.progress('Construct exact solution in matrix format', 100)

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

    print('\nNumber of snapshots: ' + str(i))
    print('-' * (21 + len(str(i))) + '\n')

    # storing index
    ind = int((i - snapshots_min) / snapshots_step)

    # generate data
    utl.progress('Generate test data', 0, dots=13)
    [x, y] = mdl.fermi_pasta_ulam_data(d, i)
    utl.progress('Generate test data', 100, dots=13)

    # computation in matrix format
    if i <= snapshots_mat:
        utl.progress('Running matrix-based MANDy', 0, dots=5)

        # construct psi_x in matrix format
        psi_x = np.zeros([p ** d, i])
        for j in range(i):
            c = [psi[l](x[0, j]) for l in range(p)]
            for k in range(1, d):
                c = np.tensordot(c, [psi[l](x[k, j]) for l in range(p)], axes=0)
            psi_x[:, j] = c.reshape(psi_x.shape[0])

        utl.progress('Running matrix-based MANDy', 50, dots=5)

        # compute xi in matrix format
        with utl.Timer() as time:
            [u, s, v] = splin.svd(psi_x, full_matrices=False, overwrite_a=True, check_finite=False,
                                  lapack_driver='gesvd')
            xi = y @ v.transpose() @ np.diag(np.reciprocal(s)) @ u.transpose()
        cpu_times[0, ind] = time.elapsed
        rel_errors[0, ind] = np.linalg.norm(xi.transpose() - xi_exact_mat) / np.linalg.norm(xi_exact_mat)
        del xi, u, s, v
        utl.progress('Running matrix-based MANDy', 100, dots=5)
        print('   CPU time      : ' + str("%.2f" % cpu_times[0, ind]) + 's')
        print('   relative error: ' + str("%.2e" % rel_errors[0, ind]))

    else:

        # extrapolate cpu times of matrix approach
        cpu_times[0, ind] = 2 * cpu_times[0, ind - 1] - cpu_times[0, ind - 2]

    # exact computation in TT format
    utl.progress('Running MANDy (eps=0)', 0, dots=10)
    with utl.Timer() as time:
        xi = mandy.mandy_cm(x, y, psi, threshold=0)
    cpu_times[1, ind] = time.elapsed
    rel_errors[1, ind] = (xi - xi_exact).norm() / xi_exact.norm()
    del xi
    utl.progress('Running MANDy (eps=0)', 100, dots=10)
    print('   CPU time      : ' + str("%.2f" % cpu_times[1, ind]) + 's')
    print('   relative error: ' + str("%.2e" % rel_errors[1, ind]))

    # use thresholds larger 0 for orthonormalizations
    for j in range(0, 4):
        utl.progress('Running MANDy (eps=10^' + str(-10 + j) + ')', 0, dots=8 - len(str(-10 + j)))
        with utl.Timer() as time:
            xi = mandy.mandy_cm(x, y, psi, threshold=10 ** (-10 + j))
        cpu_times[j + 2, ind] = time.elapsed
        rel_errors[j + 2, ind] = (xi - xi_exact).norm() / xi_exact.norm()
        del xi
        utl.progress('Running MANDy (eps=10^' + str(-10 + j) + ')', 100, dots=8 - len(str(-10 + j)))
        print('   CPU time      : ' + str("%.2f" % cpu_times[j + 2, ind]) + 's')
        print('   relative error: ' + str("%.2e" % rel_errors[j + 2, ind]))

# plot results
# ------------

utl.plot_parameters()

# CPU times
plt.figure(dpi=300)
x_values = np.arange(snapshots_min, snapshots_max + 1, snapshots_step)
plt.plot(x_values[:index_mat], cpu_times[0, :index_mat], label=r"Matrix repr.")
plt.plot(x_values[index_mat-1:], cpu_times[0, index_mat-1:], 'C0--')
plt.plot(x_values, cpu_times[1, :], label=r"TT - exact")
plt.plot(x_values, cpu_times[2, :], label=r"TT - $\varepsilon=10^{-10}$")
plt.plot(x_values, cpu_times[3, :], label=r"TT - $\varepsilon=10^{-9}$")
plt.plot(x_values, cpu_times[4, :], label=r"TT - $\varepsilon=10^{-8}$")
plt.plot(x_values, cpu_times[5, :], label=r"TT - $\varepsilon=10^{-7}$")
plt.gca().set_xlim([snapshots_min, snapshots_max])
plt.gca().set_ylim(bottom=0)
plt.title(r'CPU times for $d = 10$', y=1.03)
plt.xlabel(r'$m$')
plt.ylabel(r'$T / s$')
plt.legend(loc=2, fontsize=12).get_frame().set_alpha(1)
plt.show()

# relative errors
plt.figure(dpi=300)
x_values = np.arange(snapshots_min, snapshots_max + 1, snapshots_step)
plt.semilogy(x_values[:index_mat], rel_errors[0, :index_mat], label="Matrix - exact")
for j in range(5):
    exp = -11 + j
    if exp == -11:
        plt.semilogy(x_values[:index_mat], rel_errors[1, :index_mat], label=r'TT - exact')
    else:
        plt.semilogy(x_values[:index_mat], rel_errors[1 + j, :index_mat], label=r'TT - $\varepsilon = 10^{' + str(exp) + '}$')
plt.grid(which='major')
plt.grid(which='minor')
plt.title(r'Relative errors for $d = 10$', y=1.03)
plt.xlabel(r'$m$')
plt.ylabel(r'$|| \Xi_{\textrm{exact}} - \Xi ||~/~|| \Xi_{\textrm{exact}} ||$')
plt.gca().set_xlim([x_values[0], x_values[index_mat-1]])
plt.legend(loc=1, fontsize=12).get_frame().set_alpha(1)
plt.show()
