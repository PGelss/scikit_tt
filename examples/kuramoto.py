# -*- coding: utf-8 -*-

"""
This is an example of the application of MANDy to a high-dimensional dynamical system. See [1]_ for details.

References
----------
.. [1] P. Gelß, S. Klus, J. Eisert, C. Schütte, "Multidimensional Approximation of Nonlinear Dynamical Systems",
       arXiv:1809.02448, 2018
"""

import numpy as np
import scipy.integrate as spint
from scikit_tt.tensor_train import TT
import scikit_tt.mandy as mandy
import scikit_tt.models as mdl
import scikit_tt.utils as utl
import matplotlib.pyplot as plt
import matplotlib.patches as pat


def reconstruction():
    """Reconstruction of the dynamics of the Kuramoto model"""

    def approximated_dynamics(_, theta):
        """Construction of the right-hand side of the system from the coefficient tensor"""

        cores = [np.zeros([1, theta.shape[0] + 1, 1, 1])] + [np.zeros([1, theta.shape[0] + 1, 1, 1]) for _ in
                                                             range(1, p)]
        for q in range(p):
            cores[q][0, :, 0, 0] = [1] + [psi[q](theta[r]) for r in range(theta.shape[0])]
        psi_x = TT(cores)
        psi_x = psi_x.full().reshape(np.prod(psi_x.row_dims), 1)

        rhs = psi_x.transpose() @ xi
        rhs = rhs.reshape(rhs.size)
        return rhs

    sol = spint.solve_ivp(approximated_dynamics, [0, time], x_0, method='BDF', t_eval=np.linspace(0, time, m))
    sol = sol.y

    return sol


utl.header(title='MANDy - Kuramoto model')

# model parameters
# ----------------

# number of oscillators
d = 100

# initial distribution
x_0 = 2 * np.pi * np.random.rand(d) - np.pi

# natural frequencies
w = np.linspace(-5, 5, d)

# coupling strength
k = 2

# external forcing
h = 0.2

# basis functions
psi = [lambda t: np.sin(t), lambda t: np.cos(t)]
p = len(psi)

# integration time and number of snapshots
time = 1020
m = 10201

# threshold
threshold = 0

# recovering of the dynamics
# --------------------------

print('Recovering of the dynamics')
print('-' * 50)

# construct exact solution in TT and matrix format
utl.progress('Construct exact solution in TT format', 0)
xi_exact = mdl.kuramoto_coefficients(d, w, k, h)
utl.progress('Construct exact solution in TT format', 100)

# generate data
utl.progress('Generate test data', 0, dots=22)
[x, y] = mdl.kuramoto(x_0, w, k, h, time, m)
utl.progress('Generate test data', 100, dots=22)

# apply MANDy (function-major)
utl.progress('Running MANDy (eps=' + str(threshold) + ')', 0, dots=20 - len(str(threshold)))
with utl.Timer() as time:
    xi = mandy.mandy_fm(x, y, psi, threshold=threshold)
utl.progress('Running MANDy (eps=' + str(threshold) + ')', 100, dots=20 - len(str(threshold)))

# CPU time and relative error
print('CPU time ' + '.' * 32 + ' ' + str("%.2f" % time.elapsed))
print('Relative error ' + '.' * 26 + ' ' + str("%.2e" % ((xi - xi_exact).norm() / xi_exact.norm())))

# comparison of approximate and real dynamics
# -------------------------------------------

print('\nComparison of approximate and real dynamics')
print('-' * 50)

# convert xi to full format
xi = xi.full().reshape(np.prod(xi.row_dims[:-1]), xi.row_dims[-1])

# set new parameters
x_0 = 2 * np.pi * np.random.rand(d) - np.pi
time = 100
m = 6

# generate new data
utl.progress('\Generate test data', 0, dots=22)
[x, _] = mdl.kuramoto(x_0, w, k, h, time, m)
utl.progress('Generate test data', 100, dots=22)

# reconstruct data
utl.progress('Reconstruct test data', 0, dots=19)
x_reconstructed = reconstruction()
utl.progress('Reconstruct test data', 100, dots=19)

# relative error
print('Relative error ' + '.' * 26 + ' ' + str("%.2e" % (np.linalg.norm(x - x_reconstructed) / np.linalg.norm(x))))
print(' ')

# plot results
# ------------

utl.plot_parameters()

pos = np.zeros([2, 10])
pos2 = np.zeros([2, 10])

plt.figure(dpi=300)

for k in range(m):
    plt.subplot(2, 3, k + 1)

    for i in range(0, 10):
        pos[0, i] = np.cos(x[i * int(d / 10), k])
        pos[1, i] = np.sin(x[i * int(d / 10), k])
        pos2[0, i] = np.cos(x_reconstructed[i * int(d / 10), k])
        pos2[1, i] = np.sin(x_reconstructed[i * int(d / 10), k])

    circ = pat.Circle((0, 0), radius=1, edgecolor='gray', facecolor='aliceblue')
    plt.gca().add_patch(circ)
    for i in range(10):
        plt.plot(pos[0, i], pos[1, i], 'o', markersize=10, markeredgecolor='C' + str(i), markerfacecolor='white')
        plt.plot([0, pos2[0, i]], [0, pos2[1, i]], '--', color='gray', linewidth=0.5)
        plt.plot(pos2[0, i], pos2[1, i], 'o', markeredgecolor='C' + str(i), markerfacecolor='C' + str(i))

    plt.gca().set_xlim([-1.1, 1.1])
    plt.gca().set_ylim([-1.1, 1.1])
    plt.gca().set_aspect('equal')
    plt.axis('off')
    plt.title(r't=' + str(k * 20) + 's', y=1, fontsize=12)
plt.show()
