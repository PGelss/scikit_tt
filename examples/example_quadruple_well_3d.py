#!/usr/bin/env python3
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
import scipy.io as spio
import scipy.sparse.linalg as splin
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import scikit_tt as tt
import solvers.EVP as evp
import tools.tools as tls
import os

# parameters
# ----------

snapshots = 100
dimension = 25
number_ev = 3

# load data obtained by applying Ulam's method
# --------------------------------------------

tls.progress('\nLoad data', 0, dots=39)
ind = spio.loadmat(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/data/QuadrupleWell3D_25x25x25_100.mat")["indices"]  # load data
tls.progress('Load data', 100, dots=39)

# construct TT operator
# ---------------------

cores = [None] * 3

# find unique indices for transitions in the first dimension
ind1_unique, ind1_inv = np.unique(ind[[0, 3], :], axis=1, return_inverse=True)
rank1 = ind1_unique.shape[1]
cores[0] = np.zeros([1, dimension, dimension, rank1])
for i in range(rank1):
    cores[0][0, ind1_unique[0, i] - 1, ind1_unique[1, i] - 1, i] = 1

# find unique indices for transitions in the third dimension
ind2_unique, ind2_inv = np.unique(ind[[2, 5], :], axis=1, return_inverse=True)
rank2 = ind2_unique.shape[1]
cores[2] = np.zeros([rank2, dimension, dimension, 1])
for i in range(rank2):
    cores[2][i, ind2_unique[0, i] - 1, ind2_unique[1, i] - 1, 0] = 1

# construct core for the second dimension
cores[1] = np.zeros([rank1, dimension, dimension, rank2])
for i in range(snapshots * dimension ** 3):
    tls.progress('Construct operator', 100 * i / (snapshots * dimension ** 3 - 1), dots=30)
    cores[1][ind1_inv[i], ind[1, i] - 1, ind[4, i] - 1, ind2_inv[i]] += 1

# transpose and normalize operator
operator = (1 / snapshots) * tt.TT(cores).transpose()

# approximate leading eigenfunctions
# ----------------------------------

tls.progress('Approximate eigenfunctions in the TT format', 0, dots=5)
initial = tt.TT.uniform(operator.row_dims, ranks=[1, 20, 10, 1])
eigenfunctions, eigenvalues = evp.als(operator, initial, number_ev, 2)
tls.progress('Approximate eigenfunctions in the TT format', 100, dots=5)

# compute exact eigenvectors
# --------------------------

tls.progress('Compute exact eigenfunctions in matrix format', 0)
eigenvalues_exact, eigenfunctions_exact = splin.eigs(operator.matricize(), k=number_ev)
idx = eigenvalues_exact.argsort()[::-1]
eigenvalues_exact = eigenvalues_exact[idx]
eigenfunctions_exact = eigenfunctions_exact[:, idx]
tls.progress('Compute exact eigenfunctions in matrix format', 100)

# reshape and normalize eigentensors
# ----------------------------------

eigentensors_tt = [None] * number_ev
eigentensors_ex = [None] * number_ev
for i in range(number_ev):
    eigentensors_tt[i] = tt.TT.full(eigenfunctions[i])[:, :, :, 0, 0, 0]
    eigentensors_ex[i] = eigenfunctions_exact[:, i].reshape(dimension, dimension, dimension)
    if i == 0:
        eigentensors_tt[i] = eigentensors_tt[i] / np.sum(eigentensors_tt[i])
        eigentensors_ex[i] = eigentensors_ex[i] / np.sum(eigentensors_ex[i])
    else:
        eigentensors_tt[i] = eigentensors_tt[i] / np.amax(abs(eigentensors_tt[i]))
        eigentensors_ex[i] = eigentensors_ex[i] / np.amax(abs(eigentensors_ex[i]))

# print errors
# ------------

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


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'figure.autolayout': True})
f = plt.figure(figsize=plt.figaspect(0.4))
for i in range(number_ev):
    indices = np.where(abs(eigentensors_tt[i]/np.amax(abs(eigentensors_tt[i]))) > 0.001)
    ax = f.add_subplot(1, 3, i + 1, projection='3d', aspect=1)
    im = ax.scatter(indices[0], indices[1], indices[2], c=eigentensors_tt[i][indices], cmap='jet',
                    s=abs(eigentensors_tt[i])[indices]/np.amax(abs(eigentensors_tt[i])) * 100, vmin=np.amin(eigentensors_tt[i]),
                    vmax=np.amax(eigentensors_tt[i]))
    f.colorbar(im, shrink=0.4, aspect=10)
    ax.set_title(r'$\lambda$=' + str("%.4f" % np.abs(eigenvalues_exact[i])))
    ax.xaxis.set_ticklabels([])
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticklabels([])
    ax.yaxis.set_ticks([])
    ax.zaxis.set_ticklabels([])
    ax.zaxis.set_ticks([])
plt.suptitle('Eigenfunctions of the Perron-Frobenius operator', fontsize=18, y=0.925)
plt.tight_layout()
plt.show()
