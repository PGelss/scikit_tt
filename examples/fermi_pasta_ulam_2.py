# -*- coding: utf-8 -*-

"""
This is an example of the application of MANDy to a high-dimensional dynamical system. See [1]_ for details.

References
----------
.. [1] P. Gelß, S. Klus, J. Eisert, C. Schütte, "Multidimensional Approximation of Nonlinear Dynamical Systems",
       arXiv:1809.02448, 2018
"""

import numpy as np
import scikit_tt.data_driven.mandy as mandy
import scikit_tt.models as mdl
import scikit_tt.utils as utl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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
    utl.progress('Construct exact solution in TT format', 0, dots=3)
    xi_exact = mdl.fpu_coefficients(i)
    utl.progress('Construct exact solution in TT format', 100, dots=3)

    # generate data
    utl.progress('Generate test data', 0, dots=22)
    [x, y] = mdl.fermi_pasta_ulam(i, snapshots_max)
    utl.progress('Generate test data', 100, dots=22)

    for j in range(snapshots_min, snapshots_max + snapshots_step, snapshots_step):
        # storing indices
        ind_1 = rel_errors.shape[0] - 1 - int((j - snapshots_min) / snapshots_step)
        ind_2 = int((i - d_min))

        utl.progress('Running MANDy', 100 * (rel_errors.shape[0] - ind_1 - 1) / rel_errors.shape[0], dots=27)

        # approximate coefficient tensor
        xi = mandy.mandy_cm(x[:, :j], y[:, :j], psi, threshold=1e-10)
        rel_errors[ind_1, ind_2] = (xi - xi_exact).norm() / xi_exact.norm()
        del xi

    utl.progress('Running MANDy', 100, dots=27)
    print(' ')

# plot results
# ------------

utl.plot_parameters()
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
