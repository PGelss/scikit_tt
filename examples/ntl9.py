# -*- coding: utf-8 -*-

"""
This is an example of tensor-based EDMD. See [1]_ for details.

References
----------
..[1] F. Nüske, P. Gelß, S. Klus, C. Clementi. "Tensor-based EDMD for the Koopman analysis of high-dimensional
          systems", arXiv:1908.04741, 2019
"""

import time as _time

import matplotlib.pyplot as plt
import numpy as np

import scikit_tt.data_driven.tedmd as tedmd
import scikit_tt.data_driven.transform as tdt
import scikit_tt.utils as utl


def load_data(path, downsampling_rate, contact_indices, progress=True):
    """Load trajectory data.

    Parameters
    ----------
    path: string
        path where the files are stored
    downsampling_rate: int
        only consider trajectories numbers 0, downsampling_rate, 2*downsampling_rate, ...
    contact_indices: ndarray of ints
        only extract a given subset of indices

    Returns
    -------
    data: ndarray
        data matrix
    trajectory_lengths: list of ints
        number of snapshots in each trajectory
    """

    # define list of files
    files = [path + "Contact_Traj_%d.npz" % i for i in range(4)]

    # load data
    start_time = utl.progress('Load trajectory data', 0, show=progress)
    m = 0
    data = []
    trajectory_lengths = []
    for i in range(4):
        trajectory = np.load(files[i])["feature_traj"][contact_indices, ::downsampling_rate]
        data.append(trajectory)
        trajectory_lengths.append(trajectory.shape[1])
        m += trajectory.shape[1]
        utl.progress('Load trajectory data (m=' + str(m) + ')', 100 * (i + 1) / 4, cpu_time=_time.time() - start_time,
                     show=progress)
    data = np.hstack(data)

    return data, trajectory_lengths


def xy_indices(trajectory_lengths: list, integer_lag_time):
    """Select snapshot indices for x and y data corresponding to given integer lag time.

    Parameters
    ----------
    trajectory_lengths: list of integers
        list of trajectory lengths
    integer_lag_time: int
        integer lag time

    Returns
    ----------
    x_indices: ndarray of ints
        indices of snapshots that form x
    y_indices: ndarray of ints
        indices of snapshots that form y
    """

    # define x and y index arrays
    x_indices = np.array([], dtype=int)
    y_indices = np.array([], dtype=int)

    # Loop over trajectories:
    pos = 0
    for i in range(len(trajectory_lengths)):
        x_indices = np.concatenate((x_indices, np.arange(pos, pos + trajectory_lengths[i] - integer_lag_time)))
        y_indices = np.concatenate((y_indices, np.arange(pos + integer_lag_time, pos + trajectory_lengths[i])))
        pos += trajectory_lengths[i]

    return x_indices, y_indices


def plot_basis_functions(dimension, downsampling_rate, directory):
    """Plot basis functions

    Parameters
    ----------
    dimension: int
        number of contact indices
    downsampling_rate: int
        downsampling rate for trajectory data
    directory: string
        directory data to load
    """

    # load contact indices (sorted by relevance)
    contact_indices = np.load(directory + 'ntl9_contact_indices.npz')['indices'][:dimension]

    # load trajectories
    data, trajectory_lengths = load_data(directory, 1, contact_indices)

    # plot basis functions
    plt.figure(dpi=300)
    plt.hist(data[:10].flatten(), 10000, histtype='bar')
    x_values = np.arange(0, 1, 0.001)[:, None]
    y_values = tdt.GaussFunction(0, 0.285, 0.001)(x_values.T)
    plt.plot(x_values, 170000 * y_values, linewidth=2)
    y_values = tdt.GaussFunction(0, 0.62, 0.01)(x_values.T)
    plt.plot(x_values, 2500 * y_values, linewidth=2)
    plt.xlim([0, 1])
    plt.xlabel(r"$\Delta [\mathrm{nm}]$")
    plt.savefig(directory + 'ntl9_basis_functions_1.pdf')
    plt.show()
    plt.figure(dpi=300)
    x = np.arange(0, 1, 0.001)[:, None]
    plt.plot(x, tdt.ConstantFunction(0)(x.T), linewidth=2)
    plt.plot(x, tdt.GaussFunction(0, 0.285, 0.001)(x.T), linewidth=2)
    plt.plot(x, tdt.GaussFunction(0, 0.62, 0.01)(x.T), linewidth=2)
    plt.xlim([0, 1])
    plt.ylim([0, 1.2])
    plt.xlabel(r"$\Delta [\mathrm{nm}]$")
    plt.savefig('ntl9_basis_functions_2.pdf')
    plt.show()


def tedmd_hosvd(dimensions, downsampling_rate, integer_lag_times, threshold, directory):
    """tEDMD using AMUSEt with HOSVD

    Parameters
    ----------
    dimensions: list[int]
        numbers of contact indices
    downsampling_rate: int
        downsampling rate for trajectory data
    integer_lag_times: list[int]
        integer lag times for application of tEDMD
    threshold: float
        threshold for SVD/HOSVD
    directory: string
        directory data to load
    """


    for i in range(len(dimensions)):

        # progress
        if i==0:
             start_time = utl.progress('Apply AMUSEt (HOSVD, p=' + str(dimensions[i]) + ')', 0)

        # parameters
        time_step = 2e-3
        lag_times = time_step * downsampling_rate * integer_lag_times

        # define basis list
        basis_list = [[tdt.ConstantFunction(i), tdt.GaussFunction(i, 0.285, 0.001), tdt.GaussFunction(i, 0.62, 0.01)] for
                      i in range(dimensions[i])]

        # load contact indices (sorted by relevance)
        contact_indices = np.load(directory + 'ntl9_contact_indices.npz')['indices'][:dimensions[i]]

        # load data
        data, trajectory_lengths = load_data(directory, downsampling_rate, contact_indices, progress=False)

        # select snapshot indices for x and y data
        x_indices = []
        y_indices = []
        for j in range(len(integer_lag_times)):
            x_ind, y_ind = xy_indices(trajectory_lengths, integer_lag_times[j])
            x_indices.append(x_ind)
            y_indices.append(y_ind)

        # apply AMUSEt
        with utl.timer() as timer:
            eigenvalues, _ = tedmd.amuset_hosvd(data, x_indices, y_indices, basis_list, threshold=threshold)
        cpu_time = timer.elapsed

        for j in range(len(integer_lag_times)):
            eigenvalues[j] = [eigenvalues[j][1]]

        # Save results to file:
        dic = {}
        dic["lag_times"] = lag_times
        dic["eigenvalues"] = eigenvalues
        dic["cpu_time"] = cpu_time
        np.savez_compressed(directory + "Results_NTL9_HOSVD_d" + str(dimensions[i]) + ".npz", **dic)

        # progress
        utl.progress('Apply AMUSEt (HOSVD, p=' + str(dimensions[i]) + ')', 100 * (i+1) / len(dimensions),
                     cpu_time=_time.time() - start_time)



def tedmd_hocur(dimensions, downsampling_rate, integer_lag_times, max_rank, directory):
    """tEDMD using AMUSEt with HOSVD

    Parameters
    ----------
    dimensions: list[int]
        numbers of contact indices
    downsampling_rate: int
        downsampling rate for trajectory data
    integer_lag_times: list[int]
        integer lag times for application of tEDMD
    max_rank: int
        maximum rank for HOCUR
    directory: string
        directory data to load
    """

    for i in range(len(dimensions)):

        if i==0:
            # progress
            start_time = utl.progress('Apply AMUSEt (HOCUR, p=' + str(dimensions[i]) + ')', 0)

        # parameters
        time_step = 2e-3
        lag_times = time_step * downsampling_rate * integer_lag_times

        # define basis list
        basis_list = [[tdt.ConstantFunction(i), tdt.GaussFunction(i, 0.285, 0.001), tdt.GaussFunction(i, 0.62, 0.01)] for
                      i in range(dimensions[i])]

        # load contact indices (sorted by relevance)
        contact_indices = np.load(directory + 'ntl9_contact_indices.npz')['indices'][:dimensions[i]]

        # load data
        data, trajectory_lengths = load_data(directory, downsampling_rate, contact_indices, progress=False)

        # select snapshot indices for x and y data
        x_indices = []
        y_indices = []
        for j in range(len(integer_lag_times)):
            x_ind, y_ind = xy_indices(trajectory_lengths, integer_lag_times[j])
            x_indices.append(x_ind)
            y_indices.append(y_ind)

        # apply AMUSEt
        with utl.timer() as timer:
            eigenvalues, _ = tedmd.amuset_hocur(data, x_indices, y_indices, basis_list, max_rank=max_rank)
        cpu_time = timer.elapsed

        for j in range(len(integer_lag_times)):
            eigenvalues[j] = [eigenvalues[j][1]]

        # Save results to file:
        dic = {}
        dic["lag_times"] = lag_times
        dic["eigenvalues"] = eigenvalues
        dic["cpu_time"] = cpu_time
        np.savez_compressed(directory + "Results_NTL9_HOCUR_d" + str(dimensions[i]) + ".npz", **dic)

        utl.progress('Apply AMUSEt (HOCUR)', 100 * (i + 1) / len(dimensions), cpu_time=_time.time() - start_time)


# title
utl.header(title='NTL9')

# set plot parameters
plt.rc('text', usetex=True)
plt.rc('font', family='sans')
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'figure.autolayout': True})
plt.rcParams.update({'axes.grid': True})

# data directory (3GB trajectory data, TICA results, etc. not included)
directory = '/srv/public/data/ntl9/'

# plot basis functions
dimension = 666
downsampling_rate = 1
plot_basis_functions(dimension, downsampling_rate, directory)

# tEDMD using AMUSEt with HOSVD
dimensions = [10, 20]
downsampling_rate = 25
integer_lag_times = np.array([1, 4, 10, 40, 100, 400])
threshold = 1e-3
tedmd_hosvd(dimensions, downsampling_rate, integer_lag_times, threshold, directory)

# tEDMD using AMUSEt with HOCUR
dimensions = [10, 20]
downsampling_rate = 1
integer_lag_times = np.array([10, 40, 100, 400, 1000, 4000, 10000])
max_rank = 1000
tedmd_hocur(dimensions, downsampling_rate, integer_lag_times, max_rank, directory)

# plot timescales
plt.figure(dpi=300)

# MSM result (computed with PyEMMA)
timescale_msm = 21.9152
plt.loglog([10 ** -2, 40], timescale_msm * np.ones(2), "k--", label="MSM", linewidth=2)
for i in [20, 200, 600]:
    # TICA results (computed with PyEMMA)
    tica_data = np.load(directory + "tica_p_" + str(i) + ".npz")
    lagtimes_tica = tica_data['lags']
    timescales_tica = tica_data['ts']
    plt.loglog(lagtimes_tica, timescales_tica[:, 0], "--", label=r"TICA, $p=" + str(i) + "$", linewidth=2)
for i in [10, 20]:
    hosvd_data = np.load(directory + 'Results_NTL9_HOSVD_d' + str(i) + '.npz')
    lagtimes_hosvd = hosvd_data["lag_times"]
    ev_hosvd = hosvd_data["eigenvalues"]
    print('CPU time of HOSVD approach (p=' + str(i) + '): ' + str(hosvd_data['cpu_time']) + '\n')
    timescales_hosvd = -lagtimes_hosvd[:, None] / np.log(ev_hosvd)
    plt.loglog(lagtimes_hosvd, timescales_hosvd[:, 0], "o-", label=r"HOSVD, $p=" + str(i) + "$", linewidth=2)
for i in [10, 20]:
    hocur_data = np.load(directory + 'Results_NTL9_HOCUR_d' + str(i) + '.npz')
    lagtimes_hocur = hocur_data["lag_times"]
    ev_hocur = hocur_data["eigenvalues"]
    print('CPU time of HOCUR approach (p=' + str(i) + '): ' + str(hocur_data['cpu_time']) + '\n')
    timescales_hocur = -lagtimes_hocur[:, None] / np.log(ev_hocur)
    plt.loglog(lagtimes_hocur, timescales_hocur[:, 0], "o-", label=r"HOCUR, $p=" + str(i) + "$", linewidth=2)
plt.xlabel(r"$\tau\,[\mathrm{\mu s}]$")
plt.ylabel(r"$t_2$")
plt.legend(loc=4, ncol=2, fontsize=15)
plt.xlim([10 ** -2, 40])
plt.ylim([10 ** -1, 30])
plt.savefig('ntl9_timescales.pdf')
plt.show()

# contact matrices (computed with msm_tools)
contact_mat_hocur = np.load(directory + "Contact_Matrices_HOCUR_p20.npz")
contact_mat_msm = np.load(directory + "Contact_Matrices_MSM.npz")

# plot unfolded state
plt.rcParams.update({'axes.grid': False})
plt.figure(dpi=300)
cm = contact_mat_hocur["State_0"]
cm += contact_mat_msm["State_2"]
plt.imshow(np.flipud(cm), aspect=1, vmin=0.0, vmax=1.0)
plt.colorbar()
plt.xlabel('residue number')
plt.ylabel('residue number')
plt.ylim([-0.5,38.5])
plt.savefig('ntl9_unfolded_state.pdf')
plt.show()

# plot folded state
plt.figure(dpi=300)
cm = contact_mat_hocur["State_1"]
cm += contact_mat_msm["State_3"]
plt.imshow(np.flipud(cm), aspect=1, vmin=0.0, vmax=1.0)
plt.colorbar()
plt.xlabel('residue number')
plt.ylabel('residue number')
plt.ylim([-0.5,38.5])
plt.savefig('ntl9_folded_state.pdf')
plt.show()
