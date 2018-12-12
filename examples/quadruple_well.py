# -*- coding: utf-8 -*-

"""
This is a three-dimensional example for the approximation of the Perron-Frobenius operator using the TT format.
For more details, see [1]_.

References
----------
..[1] P. Gelß. "The Tensor-Train Format and Its Applications: Modeling and Analysis of Chemical Reaction 
      Networks, Catalytic Processes, Fluid Flows, and Brownian Dynamics", Freie Universität Berlin, 2017
"""

import numpy as np
import scipy.sparse.linalg as splin
from scikit_tt.tensor_train import TT
import scikit_tt.tensor_train as tt
import scikit_tt.perron_frobenius as pf
import scikit_tt.solvers.evp as evp
import scikit_tt.utils as utl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as io

utl.header(title='Quadruple-well potential')

# parameters
# ----------

simulations = 100
n_states = 25
number_ev = 3

# load data obtained by applying Ulam's method
# --------------------------------------------


utl.progress('Load data', 0, dots=39)
transitions = io.loadmat("data/QuadrupleWell3D_25x25x25_100.mat")["indices"]  # load data
utl.progress('Load data', 100, dots=39)

# construct TT operator
# ---------------------

utl.progress('Construct operator', 0, dots=30)
operator = pf.perron_frobenius_3d(transitions, [n_states] * 3, simulations)
utl.progress('Construct operator', 100, dots=30)

# approximate leading eigenfunctions
# ----------------------------------

utl.progress('Approximate eigenfunctions in the TT format', 0, dots=5)
initial = tt.uniform(operator.row_dims, ranks=[1, 20, 10, 1])
eigenvalues, eigenfunctions = evp.als(operator, initial, number_ev, 2)
utl.progress('Approximate eigenfunctions in the TT format', 100, dots=5)

# compute exact eigenvectors
# --------------------------

utl.progress('Compute exact eigenfunctions in matrix format', 0)
eigenvalues_exact, eigenfunctions_exact = splin.eigs(operator.matricize(), k=number_ev)
idx = eigenvalues_exact.argsort()[::-1]
eigenvalues_exact = eigenvalues_exact[idx]
eigenfunctions_exact = eigenfunctions_exact[:, idx]
utl.progress('Compute exact eigenfunctions in matrix format', 100)

# convert results to matrices
# ----------------------------------

eigentensors_tt = []
eigentensors_ex = []
for i in range(number_ev):
    eigentensors_tt.append(TT.full(eigenfunctions[i])[:, :, :, 0, 0, 0])
    eigentensors_ex.append(eigenfunctions_exact[:, i].reshape(n_states, n_states, n_states))
    if i == 0:
        eigentensors_tt[i] = eigentensors_tt[i] / np.sum(eigentensors_tt[i])
        eigentensors_ex[i] = eigentensors_ex[i] / np.sum(eigentensors_ex[i])
    else:
        eigentensors_tt[i] = eigentensors_tt[i] / np.amax(abs(eigentensors_tt[i]))
        eigentensors_ex[i] = eigentensors_ex[i] / np.amax(abs(eigentensors_ex[i]))

# print table
# -----------

print('\n---------------------------------------------')
print('k    lambda_k    err(lambda_k)    err(phi_k))')
print('---------------------------------------------')
for i in range(number_ev):
    number = str(i + 1)
    eigenvalue = str("%.4f" % np.abs(eigenvalues_exact[i]))
    err_eigenvalue = str("%.2e" % (np.abs(eigenvalues_exact[i] - eigenvalues[i]) / np.abs(eigenvalues_exact[i])))
    err_eigentensor = str("%.2e" % (np.amin([np.linalg.norm(eigentensors_tt[i] - eigentensors_ex[i]),
                                             np.linalg.norm(eigentensors_tt[i] + eigentensors_ex[i])]) / np.linalg.norm(
        eigentensors_ex[i])))
    print(number + 5 * ' ' + eigenvalue + 8 * ' ' + err_eigenvalue + 9 * ' ' + err_eigentensor)
print('---------------------------------------------\n')

# plot eigenfunctions
# -------------------

utl.plot_parameters()
f = plt.figure(figsize=plt.figaspect(0.35))
for i in range(number_ev):
    indices = np.where(abs(eigentensors_tt[i] / np.amax(abs(eigentensors_tt[i]))) > 0.001)
    ax = f.add_subplot(1, 3, i + 1, projection='3d', aspect=1)
    if i == 0:
        im = ax.scatter(indices[0], indices[1], indices[2], c=eigentensors_tt[i][indices], cmap='jet',
                        s=abs(eigentensors_tt[i])[indices] / np.amax(abs(eigentensors_tt[i])) * 100,
                        vmin=np.amin(eigentensors_tt[i]), vmax=np.amax(eigentensors_tt[i]))
    else:
        im = ax.scatter(indices[0], indices[1], indices[2], c=eigentensors_tt[i][indices], cmap='jet',
                        s=abs(eigentensors_tt[i])[indices] / np.amax(abs(eigentensors_tt[i])) * 100,
                        vmin=-1, vmax=1)
    f.colorbar(im, shrink=0.4, aspect=10)
    ax.set_title(r'$\lambda$=' + str("%.4f" % np.abs(eigenvalues_exact[i])), fontsize=14)
    ax.xaxis.set_ticklabels([])
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticklabels([])
    ax.yaxis.set_ticks([])
    ax.zaxis.set_ticklabels([])
    ax.zaxis.set_ticks([])
plt.suptitle('Eigenfunctions of the Perron-Frobenius operator', fontsize=25)
plt.tight_layout()
plt.show()
