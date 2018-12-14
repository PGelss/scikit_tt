# -*- coding: utf-8 -*-

"""
This is an example of the application of MANDy to a high-dimensional dynamical system. See [1]_ for details.

References
----------
.. [1] P. Gelß, S. Klus, J. Eisert, C. Schütte, "Multidimensional Approximation of Nonlinear Dynamical Systems",
       arXiv:1809.02448, 2018
"""

import numpy as np
from scikit_tt.tensor_train import TT
import scikit_tt.mandy as mandy
import scikit_tt.models as mdl
import scikit_tt.utils as utl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def fpu_coefficient_tensor(d):
    """Construction of the exact solution of the Fermi-Pasta-Ulam model in TT format. See [1]_ for details.

    Parameters
    ----------
    d: int
        number of oscillators

    Returns
    -------
    coefficient_tensor: instance of TT class
        exact coefficient tensor

    References
    ----------
    .. [1] P. Gelß, S. Klus, J. Eisert, C. Schütte, "Multidimensional Approximation of Nonlinear Dynamical Systems",
           arXiv:1809.02448, 2018
    """

    # define core types
    core_type_1 = np.zeros([1, 4, 1, 1])  # define core types
    core_type_1[0, 0, 0, 0] = 1
    core_type_2 = np.eye(4).reshape([1, 4, 1, 4])
    core_type_3 = np.zeros([4, 4, 1, 4])
    core_type_3[0, 1, 0, 0] = -2
    core_type_3[0, 3, 0, 0] = -1.4
    core_type_3[0, 0, 0, 1] = 1
    core_type_3[0, 2, 0, 1] = 2.1
    core_type_3[0, 1, 0, 2] = -2.1
    core_type_3[0, 0, 0, 3] = 0.7
    core_type_3[1, 0, 0, 0] = 1
    core_type_3[1, 2, 0, 0] = 2.1
    core_type_3[2, 1, 0, 0] = -2.1
    core_type_3[3, 0, 0, 0] = 0.7
    core_type_4 = np.eye(4).reshape([4, 4, 1, 1])

    # construct cores
    cores = [np.zeros([1, 4, 1, 4])]
    cores[0][0, :, :, :] = core_type_3[0, :, :, :]
    cores.append(core_type_4)
    for _ in range(2, d):
        cores.append(core_type_1)
    cores.append(np.zeros([1, d, 1, 1]))
    cores[d][0, 0, 0, 0] = 1
    coefficient_tensor = TT(cores)
    for q in range(1, d - 1):
        cores = []
        for _ in range(q - 1):
            cores.append(core_type_1)
        cores.append(core_type_2)
        cores.append(core_type_3)
        cores.append(core_type_4)
        for _ in range(q + 2, d):
            cores.append(core_type_1)
        cores.append(np.zeros([1, d, 1, 1]))
        cores[d][0, q, 0, 0] = 1
        coefficient_tensor = coefficient_tensor + TT(cores)
    cores = []
    for _ in range(d - 2):
        cores.append(core_type_1)
    cores.append(core_type_2)
    cores.append(np.zeros([4, 4, 1, 1]))
    cores[d - 1][:, :, :, 0] = core_type_3[:, :, :, 0]
    cores.append(np.zeros([1, d, 1, 1]))
    cores[d][0, d - 1, 0, 0] = 1
    coefficient_tensor = coefficient_tensor + TT(cores)

    return coefficient_tensor


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
    xi_exact = fpu_coefficient_tensor(i)
    utl.progress('Construct exact solution in TT format', 100, dots=3)

    # generate data
    utl.progress('Generate test data', 0, dots=22)
    [x, y] = mdl.fermi_pasta_ulam(i, snapshots_max)
    utl.progress('Generate test data', 100, dots=22)

    for j in range(snapshots_min, snapshots_max + snapshots_step, snapshots_step):
        # storing indices
        ind_1 = rel_errors.shape[0] - 1 - int((j - snapshots_min) / snapshots_step)
        ind_2 = int((i - d_min))

        # approximate coefficient tensor
        utl.progress('Running MANDy (m=' + str(j) + ')', 0, dots=22 - len(str(j)))
        xi = mandy.mandy_cm(x[:, :j], y[:, :j], psi, threshold=1e-10)
        utl.progress('Running MANDy (m=' + str(j) + ')', 100, dots=22 - len(str(j)))
        rel_errors[ind_1, ind_2] = (xi - xi_exact).norm() / xi_exact.norm()
        del xi

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
