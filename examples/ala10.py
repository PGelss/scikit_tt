# -*- coding: utf-8 -*-

"""
This is an example of tensor-based EDMD. See [1]_ for details.

References
----------
..[1] F. Nüske, P. Gelß, S. Klus, C. Clementi. "Tensor-based EDMD for the Koopman analysis of high-dimensional
      systems", arXiv:1908.04741, 2019
"""

import numpy as np
import scikit_tt.utils as utl
import scikit_tt.data_driven.transform as tdt
import scikit_tt.data_driven.tedmd as tedmd
import matplotlib.pyplot as plt
import time as _time

def get_index_lists(trajectory_lengths, lag_times_int):
    """Compute index lists for given lag times.

    Parameters
    ----------
    trajectory_length: list of integers
        list of trajectory lengths
    lag_times_int: list of ints
        list of integer lag times

    Returns
    -------
    x_list: list of ndarrays of type int
        index list of snapshots for matrix x
    y_list: list of ndarrays of type int
        index list of snapshots for matrix y
    """

    x_list = []
    y_list = []

    # loop over lag times
    for i in range(len(lag_times_int)):

        x_indices = np.array([], dtype=int)
        y_indices = np.array([], dtype=int)

        # loop over trajectories
        position = 0
        for j in range(len(trajectory_lengths)):
            x_indices = np.concatenate((x_indices, np.arange(position, position + trajectory_lengths[j] - lag_times_int[i])))
            y_indices = np.concatenate((y_indices, np.arange(position + lag_times_int[i], position + trajectory_lengths[j])))
            position += trajectory_lengths[j]

        x_list.append(x_indices)
        y_list.append(y_indices)

    return x_list, y_list

def load_data(directory, downsampling_rate):
    """Load trajectory data for Deca-Alanine.

    Parameters
    ----------
    directory: string
        directory where the data is stored
    downsampling_rate: int
        downsampling rate for trajectory data

    Returns
    -------
    data: ndarray
        dihedral time series data
    trajectory_lengths: list of ints
        length of each trajectory
    """

    data = []
    trajectory_lengths = []
    number_of_trajectories = 6

    # loop over trajectories
    for i in range(number_of_trajectories):
        # exclude first and last dihedral pair and downsample data
        trajectory = np.load(directory + "DihedralTimeSeries_" + str(i) + ".npy")[::downsampling_rate, 2:12]
        trajectory_lengths.append(trajectory.shape[0])
        data.append(trajectory.T)
    data = np.hstack(data)

    return data, trajectory_lengths

# title
utl.header(title='Deca-alanine')

# define basis functions
basis_list = []
for i in range(5):
    basis_list.append([tdt.ConstantFunction(2 * i), tdt.PeriodicGaussFunction(2 * i, -2, 0.8),
                       tdt.PeriodicGaussFunction(2 * i, 1, 0.5)])
    basis_list.append([tdt.ConstantFunction(2 * i + 1), tdt.PeriodicGaussFunction(2 * i + 1, -0.5, 0.8),
                       tdt.PeriodicGaussFunction(2 * i + 1, 0, 4), tdt.PeriodicGaussFunction(2 * i + 1, 2, 0.8)])

# parameters
downsampling_rate = 500
lag_times_int = np.array([1, 2, 4, 8, 10, 12])
lag_times_phy = 1e-3 * downsampling_rate * lag_times_int
lag_times_msm = 1e-3 * np.array([100, 200, 500, 1000, 2000, 4000, 6000])
eps_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
rank_list = [20, 50, 100, 200, 500]

# data directory (data not included)
directory = "/srv/public/data/ala10/"

# load data (data not included)
data, trajectory_lengths = load_data(directory, downsampling_rate)

# construct index lists
x_list, y_list = get_index_lists(trajectory_lengths, lag_times_int)

# apply tEDMD with HOSVD
timescales_hosvd = np.zeros((len(eps_list), len(lag_times_int), 3))
for i in range(len(eps_list)):
    if i==0:
        start_time = utl.progress('Apply AMUSEt (HOSVD, eps=' + str("%.0e" % eps_list[i]) + ')', 0)
    eigenvalues, _ = tedmd.amuset_hosvd(data, x_list, y_list, basis_list, threshold=eps_list[i])
    for j in range(len(lag_times_phy)):
        timescales_hosvd[i, j, :] = -lag_times_phy[j] / np.log(eigenvalues[j][1:4])
    utl.progress('Apply AMUSEt (HOSVD, eps=' + str("%.0e" % eps_list[i]) + ')', 100 * (i+1) / len(eps_list), cpu_time=_time.time() - start_time)


# apply tEDMD with HOCUR
timescales_hocur = np.zeros((len(rank_list), len(lag_times_int), 3))
for i in range(len(rank_list)):
    if i==0:
        start_time = utl.progress('Apply AMUSEt (HOSVD, eps=' + str("%.0e" % eps_list[i]) + ')', 0)
    eigenvalues, _ = tedmd.amuset_hocur(data, x_list, y_list, basis_list, max_rank=rank_list[i], multiplier=2)
    for j in range(len(lag_times_phy)):
        timescales_hocur[i, j, :] = -lag_times_phy[j] / np.log(eigenvalues[j][1:4])
    utl.progress('Apply AMUSEt (HOCUR, rank=' + str(rank_list[i]) + ')', 100 * (i+1) / len(rank_list), cpu_time=_time.time() - start_time)


# load timescales computed by MSM
timescales_msm = 1e-3 * np.load(directory + "Timescales_MSM.npy")

# plot results
# ------------

# set plot parameters
plt.rc('text', usetex=True)
plt.rc('font', family='sans')
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'axes.grid': True})

# basis set for phi
plt.figure(dpi=300)
x = np.arange(-np.pi, np.pi, 0.01)[:, None]
for i in range(len(basis_list[0])):
    plt.plot(x, basis_list[0][i](x.T), linewidth=2)
plt.xlim([-np.pi, np.pi])
plt.ylim([0.2, 1.2])
plt.xticks([-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi], [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
plt.xlabel(r"$\phi$")
plt.show()

# basis set for psi
plt.figure(dpi=300)
x = np.arange(-np.pi, np.pi, 0.01)
x = np.stack((np.zeros(x.shape[0]), x))
for i in range(len(basis_list[1])):
    plt.plot(x[1, :], basis_list[1][i](x), linewidth=2)
plt.xlim([-np.pi, np.pi])
plt.ylim([0.2, 1.2])
plt.xticks([-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi], [r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
plt.xlabel(r"$\psi$")
plt.show()

# timescales (HOSVD)
plt.figure(dpi=300)
plt.axes().set_aspect(0.44)
plt.plot(lag_times_msm[2:], timescales_msm[2:, 0], "ko--", label="MSM", linewidth=2)
x = lag_times_phy
for i in range(len(eps_list)):
    plt.plot(x, timescales_hosvd[i, :, 0], "o-", label="%.0e" % eps_list[i], linewidth=2)
plt.xlim([x[0], x[-1]])
plt.ylim([4, 12])
plt.xlabel(r"$\tau\,[\mathrm{ns}]$")
plt.ylabel(r"$t_2$")
plt.legend(loc=2, ncol=2)
plt.show()

# timescales (HOCUR)
plt.figure(dpi=300)
plt.axes().set_aspect(0.44)
plt.plot(lag_times_msm[2:], timescales_msm[2:, 0], "ko--", label="MSM", linewidth=2)
x = lag_times_phy
for i in range(len(rank_list)):
    plt.plot(x, timescales_hocur[i, :, 0], "o-", label="%d" % rank_list[i], linewidth=2)
plt.xlim([x[0], x[-1]])
plt.ylim([4, 12])
plt.xlabel(r"$\tau\,[\mathrm{ns}]$")
plt.ylabel(r"$t_2$")
plt.yticks([4, 6, 8, 10, 12], [r"4", r"6", r"8", r"10", r"12"])
plt.legend(loc=2, ncol=2)
plt.show()
