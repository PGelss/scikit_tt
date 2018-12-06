#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is a two-dimensional example for the approximation of the Perron-Frobenius operator using the TT format.
For more details, see [1]_.

References
----------
..[1] P. Gelß. "The Tensor-Train Format and Its Applications: Modeling and Analysis of Chemical Reaction 
      Networks, Catalytic Processes, Fluid Flows, and Brownian Dynamics", Freie Universität Berlin, 2017
"""

import numpy as np
from scikit_tt.tensor_train import TT
import scikit_tt.tensor_train as tt
import scikit_tt.solvers.evp as evp
import scikit_tt.utils as utl
import matplotlib.pyplot as plt
import scipy.io as io

# parameters
# ----------

simulations = 500
n_states = 50
number_ev = 3

# load data obtained by applying Ulam's method
# --------------------------------------------

utl.progress('\nLoad data', 0, dots=39)
transitions = io.loadmat("data/TripleWell2D_500.mat")["indices"]
utl.progress('Load data', 100, dots=39)

# construct TT operator
# ---------------------

utl.progress('Construct operator', 0, dots=30)
operator = utl.perron_frobenius_2d(transitions, [n_states, n_states], simulations)
utl.progress('Construct operator', 100, dots=30)

# approximate leading eigenfunctions of the Perron-Frobenius and Koopman operator
# -------------------------------------------------------------------------------

initial = tt.ones(operator.row_dims, [1] * operator.order, ranks=11)
utl.progress('Approximate eigenfunctions in the TT format', 0, dots=5)
eigenvalues_pf, eigenfunctions_pf = evp.als(operator, initial, number_ev, 3)
utl.progress('Approximate eigenfunctions in the TT format', 50, dots=5)
eigenvalues_km, eigenfunctions_km = evp.als(operator.transpose(), initial, number_ev, 2)
utl.progress('Approximate eigenfunctions in the TT format', 100, dots=5)

# compute exact eigenvectors
# --------------------------

utl.progress('Compute exact eigenfunctions in matrix format', 0)
operator = operator.matricize()
eigenvalues_pf_exact, eigenfunctions_pf_exact = np.linalg.eig(operator)
utl.progress('Compute exact eigenfunctions in matrix format', 100)

# convert results to matrices
# ---------------------------

eigenfunctions_mat = []
eigenfunctions_mat_exact = []

for i in range(number_ev):
    eigenfunctions_mat.append(np.rot90(TT.full(eigenfunctions_pf[i])[:, :, 0, 0]))
    eigenfunctions_mat_exact.append(np.rot90(np.real(eigenfunctions_pf_exact[:, i]).reshape(n_states, n_states)))
    if i > 0:
        eigenfunctions_mat[i] = eigenfunctions_mat[i] / np.amax(abs(eigenfunctions_mat[i]))
        eigenfunctions_mat_exact[i] = eigenfunctions_mat_exact[i] / np.amax(abs(eigenfunctions_mat_exact[i]))
for i in range(number_ev):
    eigenfunctions_mat.append(np.rot90(TT.full(eigenfunctions_km[i])[:, :, 0, 0]))
    eigenfunctions_mat[i + 3] = eigenfunctions_mat[i + 3] / np.amax(abs(eigenfunctions_mat[i + 3]))

# compute errors
# --------------

err_eigval = []
err_eigfun = []
for i in range(number_ev):
    err_eigval.append(np.abs(eigenvalues_pf_exact[i] - eigenvalues_pf[i]) / np.abs(eigenvalues_pf_exact[i]))
    err_eigfun.append(np.amin([np.linalg.norm(eigenfunctions_mat_exact[i] - eigenfunctions_mat[i]),
                               np.linalg.norm(eigenfunctions_mat_exact[i] + eigenfunctions_mat[i])]) / np.linalg.norm(
        eigenfunctions_mat_exact[i]))

# print table
# -----------

print('\n---------------------------------------------')
print('k    lambda_k    err(lambda_k)    err(phi_k))')
print('---------------------------------------------')
for i in range(number_ev):
    print(str(i + 1) + '     ' + str("%.4f" % np.abs(eigenvalues_pf_exact[i])) + '        ' + str(
        "%.2e" % err_eigval[i]) + '         ' + str("%.2e" % err_eigfun[i]))
print('---------------------------------------------\n')

# plot eigenfunctions and table

utl.plot_parameters()

f = plt.figure(figsize=plt.figaspect(0.65))

for i in range(number_ev * 2):
    ax = f.add_subplot(2, 3, i + 1)
    if (i == 0) or (i == 3):
        im = ax.imshow(eigenfunctions_mat[i], cmap='jet', vmin=0, vmax=np.amax(abs(eigenfunctions_mat[i])))
    else:
        im = ax.imshow(eigenfunctions_mat[i], cmap='jet', vmin=-1, vmax=1)
    cbar = f.colorbar(im, shrink=0.6, aspect=10)
    cbar.ax.tick_params(labelsize=12)
    if i < 3:
        ax.set_title(r'$\lambda$=' + str("%.4f" % np.abs(eigenvalues_pf_exact[i])), fontsize=14)
    ax.xaxis.set_ticklabels([])
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticklabels([])
    ax.yaxis.set_ticks([])
f.suptitle('Eigenfunctions of the Perron-Frobenius and Koopman operator')
plt.show()
