# -*- coding: utf-8 -*-


import numpy as np
import os
import sys
import scipy.linalg as lin
from scikit_tt.tensor_train import TT
import scikit_tt.data_driven.tdmd as tdmd
import matplotlib.pyplot as plt
import scikit_tt.utils as utl
import time as _time


# noinspection PyTupleAssignmentBalance
def dmd_exact(x_data, y_data):
    # decompose x
    u, s, v = lin.svd(x_data, full_matrices=False, overwrite_a=True, check_finite=False, lapack_driver='gesvd')

    # construct reduced matrix
    reduced_matrix = u.T @ y_data @ v.T @ np.diag(np.reciprocal(s))

    # find eigenvalues
    eigenvalues, eigenvectors = lin.eig(reduced_matrix, overwrite_a=True, check_finite=False)

    # sort eigenvalues
    ind = np.argsort(eigenvalues)[::-1]
    dmd_eigenvalues = eigenvalues[ind]

    # compute modes
    dmd_modes = y_data @ v.T @ np.diag(np.reciprocal(s)) @ eigenvectors[:, ind] @ np.diag(
        np.reciprocal(dmd_eigenvalues))

    return dmd_eigenvalues, dmd_modes


utl.header(title='TDMD - von Kármán vortex street')

# load data
path = os.path.dirname(os.path.realpath(__file__))
data = np.load(path + "/data/karman_snapshots.npz")['snapshots']
number_of_snapshots = data.shape[-1] - 1

# tensor-based approach
# ---------------------

# thresholds for orthonormalizations
thresholds = [0, 1e-7, 1e-5, 1e-3]

start_time = utl.progress('applying TDMD for different thresholds', 0)

# construct x and y tensors and convert to TT format
x = TT(data[:, :, 0:number_of_snapshots, None, None, None])
y = TT(data[:, :, 1:number_of_snapshots + 1, None, None, None])

# define lists
eigenvalues_tdmd = [None] * len(thresholds)
modes_tdmd = [None] * len(thresholds)

for i in range(len(thresholds)):
    # apply exact TDMD
    eigenvalues_tdmd[i], modes_tdmd[i] = tdmd.tdmd_exact(x, y, threshold=thresholds[i])

    # convert to full format for comparison and plotting
    modes_tdmd[i] = modes_tdmd[i].full()[:, :, :, 0, 0, 0]

    utl.progress('applying TDMD for different thresholds', 100 * (i + 1) / (len(thresholds)),
                 cpu_time=_time.time() - start_time)

# matrix-based approach
# ---------------------

start_time = utl.progress('applying classical DMD', 0)

# construct tensors
x = data[:, :, 0:number_of_snapshots].reshape(data.shape[0] * data.shape[1], number_of_snapshots)
y = data[:, :, 1:number_of_snapshots + 1].reshape(data.shape[0] * data.shape[1], number_of_snapshots)

# apply exact DMD
eigenvalues_dmd, modes_dmd = dmd_exact(x, y)

# reshape result for comparison
modes_dmd = modes_dmd.reshape([data.shape[0], data.shape[1], number_of_snapshots])

utl.progress('applying classical DMD', 100, cpu_time=_time.time() - start_time)

# select modes
modes = [3, 11, 25]

# print errors
# ------------

# print table
print('\n')
print('            |      1. mode      |      2. mode      |      3. mode     ')
print('            |   e_ev     e_md   |   e_ev     e_md   |   e_ev     e_md  ')
print('-----------------------------------------------------------------------')
for i in range(len(thresholds)):
    sys.stdout.write('eps = ' + str("%.0e" % thresholds[i]))
    for j in range(len(modes)):
        # eigenvalue and mode computed by DMD
        ev_dmd = eigenvalues_dmd[modes[j]]
        md_dmd = modes_dmd[:, :, modes[j]]

        # eigenvalue and mode computed by TDMD
        index = np.argmin(np.abs(eigenvalues_tdmd[i] - ev_dmd))
        ev_tdmd = eigenvalues_tdmd[i][index]
        md_tdmd = modes_tdmd[i][:, :, index]

        # relative errors
        error_ev = np.abs(ev_dmd - ev_tdmd) / np.abs(ev_dmd)
        error_md = np.min([np.linalg.norm(md_dmd - md_tdmd), np.linalg.norm(md_dmd + md_tdmd)]) / np.linalg.norm(md_dmd)

        sys.stdout.write(' | ' + str("%.2e" % error_ev) + ' ' + str("%.2e" % error_md))
    sys.stdout.write('\n')
print(' ')

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'axes.grid': True})
plt.rcParams.update({'axes.grid': False})
f = plt.figure(figsize=plt.figaspect(1.75))
for i in range(len(modes)):
    ax = f.add_subplot(3, 1, i + 1, aspect=0.5)
    ax.imshow(np.real(modes_dmd[:, :, modes[i]]), cmap='jet')
    plt.axis('off')
    ev = eigenvalues_dmd[modes[i]]

    plt.title(r'$\lambda ~ = ~ $' + str("%.2f" % np.real(ev)) + '+' + str("%.2f" % np.imag(ev)) + r'$i$')
plt.show()
