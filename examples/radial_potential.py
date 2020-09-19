# -*- coding: utf-8 -*-

"""
This is a low-dimensional example for the application of HOSVD, HOCUR, and tEDMD. For more details, see [1]_.

References
----------
..[1] F. Nüske, P. Gelß, S. Klus, C. Clementi. "Tensor-based EDMD for the Koopman analysis of high-dimensional systems",
      arXiv:1908.04741, 2019
"""

import os
import sys
import time as _time

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as splin
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D

import scikit_tt.data_driven.tedmd as tedmd
import scikit_tt.data_driven.transform as tdt
import scikit_tt.utils as utl
from scikit_tt.tensor_train import TT

sys.path.insert(0, "/storage/mi/gelssp/software/d3s/d3s")
# noinspection PyUnresolvedReferences
import d3s.algorithms as alg


def potential(x):
    """Two-dimensional radial potential.

    Parameters
    ----------
    x: ndarray
        two-dimensional vector

    Returns
    -------
    v: float
        potential evaluated at x
    """

    # radial
    r = np.sqrt(x[0] ** 2 + x[1] ** 2 + 0.1)

    # potential as function of r
    v = -2 * np.exp(-2 * r ** 2) - 1.5 * np.exp(-2 * (r - 2) ** 2) - 1.5 * np.exp(-2 * (r - 4) ** 2) + np.exp(r - 6)

    return v


def construct_tdm(data, basis_list):
    """Construct transformed data matrices.

    Parameters
    ----------
    data: ndarray
        snapshot matrix
    basis_list: list of lists of lambda functions
        list of basis functions in every mode

    Returns
    -------
    psi_x: instance of TT class
        transformed data matrix corresponding to x
    psi_y: instance of TT class
        transformed data matrix corresponding to y
    """

    # extract snapshots for x and y
    x = data[:, :-1]
    y = data[:, 1:]

    # construct psi_x and psi_y
    start_time = utl.progress('Construct transformed data matrices', 0)
    psi_x = tdt.basis_decomposition(x, basis_list).transpose(cores=2).matricize()
    utl.progress('Construct transformed data matrices', 50, cpu_time=_time.time() - start_time)
    psi_y = tdt.basis_decomposition(y, basis_list).transpose(cores=2).matricize()
    utl.progress('Construct transformed data matrices', 100, cpu_time=_time.time() - start_time)

    return psi_x, psi_y


def amuse(psi_x, psi_y, threshold):
    """AMUSE (matrix case).

    Parameters
    ----------
    psi_x: array
        transformed data matrix
    psi_y: array
        transformed data matrix
    threshold: float
        threshold for SVD

    Returns
    -------
    eigenvalues: array
        EDMD eigenvalues
    eigenvectors: array
        EDMD eigenvectors
    """

    start_time = utl.progress('Apply AMUSE', 0)

    # construct reduced matrix
    # noinspection PyTupleAssignmentBalance
    u, s, v = splin.svd(psi_x, full_matrices=False, overwrite_a=True, check_finite=False, lapack_driver='gesvd')
    indices = np.where(s > threshold)[0]
    u = u[:, indices]
    s = s[indices]
    v = v[indices, :]
    s_inv = np.diag(np.reciprocal(s))
    reduced_matrix = v.dot(psi_y.T).dot(u).dot(s_inv)

    utl.progress('Apply AMUSE', 50, cpu_time=_time.time() - start_time)

    # solve eigenvalue problem and compute EDMD eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(reduced_matrix)
    idx = (np.abs(eigenvalues - 1)).argsort()[:3]
    eigenvalues = np.real(eigenvalues[idx])
    eigenvectors = u.dot(s_inv).dot(np.real(eigenvectors[:, idx]))

    utl.progress('Apply AMUSE', 100, cpu_time=_time.time() - start_time)

    return eigenvalues, eigenvectors


def approximate(x, psi_x, thresholds: list, max_ranks: list):
    """Approximate psi_x using HOSVD and HOCUR, respectively.

    Parameters
    ----------
    x: array
        snapshot matrix
    psi_x: array
        transformed data matrix
    thresholds: list of floats
        tresholds for HOSVD
    max_ranks: list of ints
        maximum ranks for HOCUR

    Returns
    -------
    ranks_hosvd: list of lists of ints
        ranks of the approximations computed with HOSVD
    errors_hosvd: list of floats
        relative errors of the approximations computed with HOSVD
    ranks_hocur: list of lists of ints
        ranks of the approximations computed with HOCUR
    errors_hocur: list of floats
        relative errors of the approximations computed with HOCUR
    """

    # define returns
    ranks_hosvd = []
    errors_hosvd = []
    ranks_hocur = []
    errors_hocur = []

    start_time = utl.progress('Approximation in TT format', 0)

    # reshape psi_x into tensor
    psi_x_full = psi_x.reshape([number_of_boxes, number_of_boxes, 1, 1, 1, psi_x.shape[1]])

    # approximate psi_x using HOSVD
    for i in range(len(thresholds)):
        psi_approx = TT(psi_x_full, threshold=thresholds[i])
        ranks_hosvd.append(psi_approx.ranks)
        errors_hosvd.append(np.linalg.norm(psi_x_full - psi_approx.full()) / np.linalg.norm(psi_x_full))
        utl.progress('Approximation in TT format', 100 * (i + 1) / 6, cpu_time=_time.time() - start_time)

    # approximate psi_x using HOCUR
    for i in range(len(max_ranks)):
        psi_approx = tdt.hocur(x, basis_list, max_ranks[i], repeats=3, multiplier=100, progress=False)
        ranks_hocur.append(psi_approx.ranks)
        errors_hocur.append(np.linalg.norm(psi_x - psi_approx.transpose(cores=2).matricize()) / np.linalg.norm(psi_x))
        utl.progress('Approximation in TT format', 100 * (i + 4) / 6, cpu_time=_time.time() - start_time)

    return ranks_hosvd, errors_hosvd, ranks_hocur, errors_hocur


def amuset(z, eigenvalues_amuse, eigenvectors_amuse, thresholds: list, max_ranks: list):
    """AMUSEt (tensor case)

    Parameters
    ----------
    z: array
        snapshot matrix
    eigenvalues_amuse: array
        eigenvalues computed by AMUSE
    eigenvectors_amuse
        eigenvectors computed by AMUSE
    thresholds: list of floats
        thresholds for AMUSEt using HOSVD
    max_ranks: list of ints
        maximum ranks for AMUSEt using HOCUR

    Returns
    -------
    ev_errors_hosvd: list of arrays
        relative errors of approximate eigenvalues (HOSVD)
    et_errors_hosvd: list of arrays
        relative errors of approximate eigentensors (HOSVD)
    ev_errors_hocur: list of arrays
        relative errors of approximate eigenvalues (HOCUR)
    et_errors_hocur: list of arrays
        relative errors of approximate eigentensors (HOCUR)
    et_last_hocur: instance of TT class
        eigentensors of last HOCUR approach
    """

    # define returns
    ev_errors_hosvd = []
    et_errors_hosvd = []
    ev_errors_hocur = []
    et_errors_hocur = []

    start_time = utl.progress('Apply AMUSEt', 0)
    eigentensors = []

    for i in range(len(thresholds) - 1):

        # apply AMUSEt using HOSVD
        eigenvalues, eigentensors = tedmd.amuset_hosvd(z, np.arange(0, z.shape[1] - 1), np.arange(1, z.shape[1]),
                                                       basis_list, threshold=thresholds[i + 1])
        # matricize eigentensors
        eigentensors = eigentensors.transpose(cores=2).matricize()

        # compute relative errors of the eigenvalues
        error_list = []
        for j in range(2):
            error_list.append(np.abs(eigenvalues[j] - eigenvalues_amuse[j]) / np.abs(eigenvalues_amuse[j]))
        ev_errors_hosvd.append(np.array(error_list))

        # compute relative errors of the eigentensors
        error_list = []
        for j in range(2):
            error_list.append(np.minimum(
                np.linalg.norm(eigentensors[:, j] - eigenvectors_amuse[:, j]) / np.linalg.norm(
                    eigenvectors_amuse[:, j]),
                np.linalg.norm(eigentensors[:, j] + eigenvectors_amuse[:, j]) / np.linalg.norm(
                    eigenvectors_amuse[:, j])))
        et_errors_hosvd.append(np.array(error_list))

        utl.progress('Apply AMUSEt', 100 * (i + 1) / 5, cpu_time=_time.time() - start_time)

    for i in range(len(max_ranks)):

        # apply AMUSEt using HOCUR
        eigenvalues, eigentensors = tedmd.amuset_hocur(z, np.arange(0, z.shape[1] - 1), np.arange(1, z.shape[1]),
                                                       basis_list, max_rank=max_ranks[i], multiplier=100)

        # matricize eigentensors
        eigentensors = eigentensors.transpose(cores=2).matricize()

        # compute relative errors of the eigenvalues
        error_list = []
        for j in range(2):
            error_list.append(np.abs(eigenvalues[j] - eigenvalues_amuse[j]) / np.abs(eigenvalues_amuse[j]))
        ev_errors_hocur.append(np.array(error_list))

        # compute relative errors of the eigentensors
        error_list = []
        for j in range(2):
            error_list.append(np.minimum(
                np.linalg.norm(eigentensors[:, j] - eigenvectors_amuse[:, j]) / np.linalg.norm(
                    eigenvectors_amuse[:, j]),
                np.linalg.norm(eigentensors[:, j] + eigenvectors_amuse[:, j]) / np.linalg.norm(
                    eigenvectors_amuse[:, j])))
        et_errors_hocur.append(np.array(error_list))

        utl.progress('Apply AMUSEt', 100 * (i + 3) / 5, cpu_time=_time.time() - start_time)

    # define et_last_hocur
    et_last_hocur = eigentensors

    return ev_errors_hosvd, et_errors_hosvd, ev_errors_hocur, et_errors_hocur, et_last_hocur


def plot_potential():
    """Plot radial potential."""
    fig = plt.figure(dpi=300)
    ax = fig.gca(projection='3d')

    # plot solid part
    x = np.arange(-5, 5, 0.05)
    y = np.arange(0, 5, 0.05)
    z = np.zeros([x.shape[0], y.shape[0]])
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            z[i, j] = potential(np.array([x[i], y[j]]))
    x, y = np.meshgrid(x, y)
    ax.plot_surface(x, y, z.T, color='#1f77b4', cstride=1, rstride=1, antialiased=False)

    # plot red line
    x = np.arange(-5, 5, 0.05)
    y = np.zeros([x.shape[0]])
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        z[i] = potential(np.array([x[i], 0]))
    plt.plot(x, y, z, 'r', zorder=3)

    # plot transparent part
    x = np.arange(-5, 5, 0.05)
    y = np.arange(-5, 0, 0.05)
    z = np.zeros([x.shape[0], y.shape[0]])
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            z[i, j] = potential(np.array([x[i], y[j]]))
    x, y = np.meshgrid(x, y)
    ax.plot_surface(x, y, z.T, color='#1f77b4', alpha=0.1, cstride=1, rstride=1, antialiased=False)

    plt.xlabel(r'$x_1$', labelpad=10)
    plt.ylabel(r'$x_2$', labelpad=8)
    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)
    ax.set_zlim(-4, 4)
    ax.set_xticks([-6, 0, 6])
    ax.set_yticks([-6, 0, 6])
    ax.set_zticks([-3, 0, 3])
    ax.set_zlabel(r'$V(x)$')
    plt.show()


def plot_eigenfunction(basis_list, eigenvector):
    """Plot eigenfunctions.

    Parameters
    ----------
    basis_list: list of lists of lambda functions
        list of basis functions in every mode
    eigenvector: array
        TEDMD eigenvector
    """

    # normalize eigenvector
    eigenvector = (1 / np.max(eigenvector)) * eigenvector

    plt.rcParams.update({'axes.grid': False})
    plt.figure(dpi=300)

    # evaluate eigenfunction over domain
    grid_size = 100
    dist = 12 / grid_size
    z = np.zeros([grid_size, grid_size])
    for i in range(grid_size):
        for j in range(grid_size):
            x_tmp = np.array([-6 + i * dist, -6 + j * dist])[:, None]
            z[i, j] = tdt.basis_decomposition(x_tmp, basis_list).matricize().dot(eigenvector)

    plt.imshow(z, cmap='seismic', vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks([0, (grid_size - 1) / 2, grid_size - 1], [-6, 0, 6])
    plt.yticks([0, (grid_size - 1) / 2, grid_size - 1], [-6, 0, 6])
    plt.xlim([0, grid_size - 1])
    plt.ylim([0, grid_size - 1])
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.rcParams.update({'axes.grid': True})


# title
utl.header(title='Radial potential')

# set plot parameters
plt.rc('text', usetex=True)
plt.rc('font', family='sans')
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'axes.grid': True})

# plot potential
plot_potential()

# parameters
directory = os.path.dirname(os.path.realpath(__file__)) + '/data/'
number_of_boxes = 100
threshold = 1e-2
thresholds = [0, 1e-12, 1e-1]
max_ranks = [2400, 2375, 2350]

# load data
z = np.load(directory + 'radial_potential_snapshots.npz')['z']

# define basis functions
diam = 12 / number_of_boxes
basis_list_1 = []
basis_list_2 = []
for i in range(number_of_boxes):
    basis_list_1.append(tdt.IndicatorFunction(0, -6 + i * diam, -6 + (i + 1) * diam))
    basis_list_2.append(tdt.IndicatorFunction(1, -6 + i * diam, -6 + (i + 1) * diam))
basis_list = [basis_list_1, basis_list_2]

# construct transformed data matrices
psi_x, psi_y = construct_tdm(z, basis_list)

# print number of visited boxes
print('Number of visited boxes (in X): ' + str(np.count_nonzero(np.sum(psi_x, axis=1))) + '\n')

# apply AMUSE
print('1st experiment\n' + 85 * '=' + '\n')
eigenvalues, eigenvectors = amuse(psi_x, psi_y, threshold)

# approximate transformed data matrix using HOSVD and HOCUR
print('2nd experiment\n' + 85 * '=' + '\n')
ranks_hosvd, errors_hosvd, ranks_hocur, errors_hocur = approximate(z[:, :-1], psi_x, thresholds, max_ranks)

# print approximation errors
print(85 * '-' + '\n' + 'Method         Threshold/maximum rank          TT ranks           Approximation error' + '\n'
      + 85 * '-')
for i in range(len(thresholds)):
    str_thr = str("%.0e" % thresholds[i])
    len_thr = len(str_thr)
    str_rank = str(ranks_hosvd[i])
    len_rank = len(str_rank)
    str_error = str('%.2e' % errors_hosvd[i])
    print('HOSVD' + 17 * ' ' + str_thr + (21 - len_thr) * ' ' + str_rank + (34 - len_rank) * ' ' + str_error)
print(85 * '-')
for i in range(len(max_ranks)):
    str_rank = str(ranks_hocur[i])
    len_rank = len(str_rank)
    str_error = str("%.2e" % errors_hocur[i])
    print('HOCUR' + 17 * ' ' + str(max_ranks[i]) + 17 * ' ' + str_rank + (34 - len_rank) * ' ' + str_error)
print(85 * '-' + '\n')

# apply AMUSEt
print('3rd experiment\n' + 85 * '=' + '\n')
ev_errors_hosvd, et_errors_hosvd, ev_errors_hocur, et_errors_hocur, et_last_hocur = amuset(z, eigenvalues[:2],
                                                                                           eigenvectors[:, :2],
                                                                                           thresholds, max_ranks)

# print eigenpair errors
print(85 * '-' + '\n' + 'Method     Threshold/maximum rank        1st eigenpair             2nd eigenpair' + '\n'
      + 38 * ' ' + 'e_lambda       e_xi       e_lambda       e_xi  ' + '\n' + 85 * '-' + '\n')
for i in range(len(thresholds) - 1):
    str_threshold = str("%.0e" % thresholds[i + 1])
    sys.stdout.write('HOSVD' + 14 * ' ' + str_threshold + 14 * ' ')
    for j in range(2):
        sys.stdout.write(
            str("%.2e" % ev_errors_hosvd[i][j]) + 5 * ' ' + str(
                "%.2e" % et_errors_hosvd[i][j]) + 5 * ' ')
    sys.stdout.write('\n')
print(85 * '-')
for i in range(len(max_ranks)):
    sys.stdout.write('HOCUR' + 14 * ' ' + str(max_ranks[i]) + 15 * ' ')
    for j in range(2):
        sys.stdout.write(
            str("%.2e" % ev_errors_hocur[i][j]) + 5 * ' ' + str(
                "%.2e" % et_errors_hocur[i][j]) + 5 * ' ')
    sys.stdout.write('\n')
print(85 * '-' + '\n')

# plot eigenfunctions
plot_eigenfunction(basis_list, -et_last_hocur[:, 1])
plot_eigenfunction(basis_list, (1 / np.max(np.abs(et_last_hocur[:, 2]))) * et_last_hocur[:, 2])
