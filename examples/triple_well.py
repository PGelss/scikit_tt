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
import scipy.io as spio
import scikit_tt as tt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# parameters
# ----------

m = 500
n = 50
number_ev = 3

# load data obtained by applying Ulam's method
# --------------------------------------------

tls.progress('\nLoad data',0,dots=39)
ind = spio.loadmat("/storage/mi/gelssp/scikit_tt/data/TripleWell2D_500.mat")["indices"]  # load data
tls.progress('Load data',100,dots=39)

# construct TT operator
# ---------------------

cores = [None] * 2
tls.progress('Construct operator', 0, dots=30)

# find unique indices for transitions in the first dimension
ind_unique, ind_inv , ind_counts= np.unique(ind[[0, 2], :], axis=1, return_inverse=True, return_counts=True)
rank = ind_unique.shape[1]
cores[0] = np.zeros([1, n , n, rank])
for i in range(rank):
    cores[0][0, ind_unique[0, i] - 1, ind_unique[1, i] - 1, i] = 1

# construct core for the second dimension
cores[1] = np.zeros([rank, n,n, 1])
for i in range(m * n * n):
    cores[1][ind_inv[i], ind[1, i] - 1, ind[3, i] - 1, 0] += 1

# transpose and normalize operator
operator = (1 / m) * tt.TT(cores).transpose()
tls.progress('Construct operator', 100, dots=30)

# approximate leading eigenfunctions of the Perron-Frobenius and Koopman operator
# -------------------------------------------------------------------------------
initial = tt.TT.ones(operator.row_dims, [1] * operator.order, ranks=11)
eigenfunctions_pf, eigenvalues_pf = EVP.als(operator, initial, number_ev, 3)
eigenfunctions_km, eigenvalues_km = EVP.als(operator.transpose(), initial, number_ev, 2)

# compute exact eigenvectors
# --------------------------
operator = operator.matricize()
eigenvalues_pf_exact, eigenfunctions_pf_exact = np.linalg.eig(operator)

# plot eigenfunctions and table

plt.switch_backend('agg')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'figure.autolayout': True})
f, ax = plt.subplots(2, 3)
f.suptitle('Eigenfunctions of the Perron-Frobenius and Koopman operator')
print('\n---------------------------------------------')
print('k    lambda_k    err(lambda_k)    err(phi_k))')
print('---------------------------------------------')
for i in range(number_ev):
    eigenfunction = np.rot90(tt.TT.full(eigenfunctions_pf[i])[:, :, 0, 0])
    eigenfunction2 = np.rot90(np.real(eigenfunctions_pf_exact[:,i]).reshape(50,50))
    if i > 0:
        eigenfunction = eigenfunction / np.amax(abs(eigenfunction))
        eigenfunction2 = eigenfunction2 / np.amax(abs(eigenfunction2))
        im = ax[0, i].imshow(eigenfunction, cmap='jet', vmin=-1, vmax=1)
    else:
        im = ax[0, i].imshow(eigenfunction, cmap='jet', vmin=0, vmax=np.amax(abs(eigenfunction)))
    divider = make_axes_locatable(ax[0, i])
    cax = divider.append_axes('right', size='10%', pad=0.05)
    f.colorbar(im, cax=cax, orientation='vertical')
    ax[0, i].set_title(r'$\lambda$=' + str("%.4f" % np.abs(eigenvalues_pf_exact[i])))
    ax[0, i].xaxis.set_ticklabels([])
    ax[0, i].xaxis.set_ticks([])
    ax[0, i].yaxis.set_ticklabels([])
    ax[0, i].yaxis.set_ticks([])
    err_eigval = np.abs(eigenvalues_pf_exact[i]-eigenvalues_pf[i])/np.abs(eigenvalues_pf_exact[i])
    err_eigfun = np.amin([np.linalg.norm(eigenfunction2-eigenfunction),np.linalg.norm(eigenfunction2+eigenfunction)])/np.linalg.norm(eigenfunction2)
    print(str(i+1)+'     '+str("%.4f" % np.abs(eigenvalues_pf_exact[i]))+'        '+str("%.2e" % err_eigval)+'         '+str("%.2e" % err_eigfun))
print('---------------------------------------------\n')
for i in range(number_ev):
    eigenfunction = np.rot90(tt.TT.full(eigenfunctions_km[i])[:, :, 0, 0])
    eigenfunction = eigenfunction / np.amax(abs(eigenfunction))
    if i > 0:
        im = ax[1, i].imshow(eigenfunction, cmap='jet', vmin=-1, vmax=1)
    else:
        im = ax[1, i].imshow(eigenfunction, cmap='jet', vmin=0, vmax=np.amax(abs(eigenfunction)))
    divider = make_axes_locatable(ax[1, i])
    cax = divider.append_axes('right', size='10%', pad=0.05)
    cbar = f.colorbar(im, cax=cax, orientation='vertical')
    ax[1, i].xaxis.set_ticklabels([])
    ax[1, i].xaxis.set_ticks([])
    ax[1, i].yaxis.set_ticklabels([])
    ax[1, i].yaxis.set_ticks([])
plt.show()
