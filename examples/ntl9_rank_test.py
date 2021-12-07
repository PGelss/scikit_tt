# -*- coding: utf-8 -*-

"""
Application of tensor-based EDMD (tEDMD) to molecular dynamics simulation data of the NTL9 protein [2].
To define elementary basis sets, we select all 666 closest-heavy-atom distances in the protein (loaded
as separate files). These are ranked according to the amount of time spent in a contact state (< 0.35nm).
We then use a set of Gaussian basis functions on the first ten or twenty of these descriptors.
We use both the direct representation as well as HOCUR-based representation of the data tensor
as starting point for the method. Several choices of the maximally allowed rank are tested, and implied timescales
computed for all of these cases.

This script requires four time series of the closest-heavy-atom distances, one for each independent MD simulation.
Moreover, we also need to load the ranking of these distances according to their fraction of time in the contact
state. The files are named

Contact_Traj_*.npz          - trajectories of closest-heavy-atom distances
ntl9_contact_indices.npz    - ranking of these distances

References
----------
..[1] F. Nüske, P. Gelß, S. Klus, C. Clementi. "Tensor-based computation of metastable and coherent sets",
 Physica D: Nonlinear Phenomena, (2021)
..[2] K. LINDORFF-LARSEN, S. PIANA, R. O. DROR, D. E. SHAW, "How Fast-Folding Proteins Fold", Science (2011)
"""

import time as _time

import numpy as np

import scikit_tt.data_driven.tedmd as tedmd
import scikit_tt.data_driven.transform as tdt
import scikit_tt.utils as utl

import os


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

    # List of contact trajectory files:
    files = [path + "Contact_Traj_%d.npz" % i for i in range(4)]

    # load and downsample the data, extract trajectory lengths:
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



def tedmd_hosvd(dimensions, downsampling_rate, integer_lag_times, threshold, max_rank, directory, res_dir):
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
    max_rank : int
        maximum rank of truncated SVD
    directory: string
        directory data to load
    res_dir: string
        directory for storing results
    """


    for i in range(len(dimensions)):

        # Elemetary simulation time step:
        time_step = 2e-3
        # lag times:
        lag_times = time_step * downsampling_rate * integer_lag_times

        # define basis list, comprised of the constant and two Gaussians
        basis_list = [[tdt.ConstantFunction(i), tdt.GaussFunction(i, 0.285, 0.001), tdt.GaussFunction(i, 0.62, 0.01)] for
                      i in range(dimensions[i])]

        # load list contact indices (sorted by fraction of simulation spent in contact state)
        contact_indices = np.load(directory + 'ntl9_contact_indices.npz')['indices'][:dimensions[i]]

        # load trajectory data
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
            eigenvalues, _ = tedmd.amuset_hosvd(data, x_indices, y_indices, basis_list, threshold=threshold,
                                                max_rank=max_rank)
        cpu_time = timer.elapsed
        print("Time Elapsed for AMUSEt, ranks = %d, p = %d: %.2f"%(max_rank, dimensions[i], cpu_time))

        for j in range(len(integer_lag_times)):
            eigenvalues[j] = [eigenvalues[j][1]]

        # Save results to file:
        dic = {}
        dic["lag_times"] = lag_times
        dic["eigenvalues"] = eigenvalues
        dic["cpu_time"] = cpu_time
        #np.savez_compressed(res_dir + "Results_NTL9_HOSVD_rank" + str(max_rank) + "_d" +
        #                    str(dimensions[i]) + ".npz", **dic)


def tedmd_hocur(dimensions, downsampling_rate, integer_lag_times, max_rank, directory, res_dir):
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
    res_dir: string
        directory for storing results
    """

    for i in range(len(dimensions)):

        # Elemetary simulation time step:
        time_step = 2e-3
        # lag times:
        lag_times = time_step * downsampling_rate * integer_lag_times

        # define basis list, comprised of the constant and two Gaussians
        basis_list = [[tdt.ConstantFunction(i), tdt.GaussFunction(i, 0.285, 0.001), tdt.GaussFunction(i, 0.62, 0.01)]
                      for
                      i in range(dimensions[i])]

        # load list contact indices (sorted by fraction of simulation spent in contact state)
        contact_indices = np.load(directory + 'ntl9_contact_indices.npz')['indices'][:dimensions[i]]

        # load trajectory data
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
        print("Time Elapsed for HOCUR, ranks = %d, p = %d: %.2f" % (max_rank, dimensions[i], cpu_time))

        for j in range(len(integer_lag_times)):
            eigenvalues[j] = [eigenvalues[j][1]]

        # Save results to file:
        dic = {}
        dic["lag_times"] = lag_times
        dic["eigenvalues"] = eigenvalues
        dic["cpu_time"] = cpu_time
        #np.savez_compressed(res_dir + "Results_NTL9_HOCUR_rank" + str(max_rank) + "_d" +
        #                    str(dimensions[i]) + ".npz", **dic)



# title
utl.header(title='NTL9')


""" Data Settings:"""
# Replace this line by the location of the contact trajectories used for the calculation (not included):
directory = "/Users/fkn1/Documents/Uni/Data/NTL9/"
# Replace this line by the directory where you would like the results to be stored:
res_dir = "/Users/fkn1/Documents/Uni/Data/21_PhysD_tEDMD_Paper/NTL9/Results/"
# Dimension of full distance coordinate set:
dimension = 666
# Downsampling parameter:
downsampling_rate = 25

""" Computational Settings: """
# Number of distances to be used with tEDMD:
dimensions = [10, 20]
# List of ranks for direct decomposition or HOCUR:
ranks = [100, 200, 500, 1000, 2000, 3000]
# Variable job_id contains the index of the maximal rank from rank_list to be used.
# If you're running this on a cluster, uncomment the next line to retrieve job_id from
# environment variables:
job_id = 1
#job_id = int(os.getenv("SLURM_ARRAY_TASK_ID")) - 1
max_rank = ranks[job_id]

# Lag times for tEDMD, defined by physical times:
lag_times_phy = 2e-3 * np.array([25, 50, 100, 250, 500, 1000, 2500, 5000])
# Convert to integer lag times:
integer_lag_times = ((1.0 / (2e-3 * downsampling_rate)) * lag_times_phy).astype(int)
threshold = 0.0
print("Physical lag times:")
print(lag_times_phy)
print("Integer lag times:")
print(integer_lag_times)

""" Run tEDMD Calculations:"""
# Run AMUSEt:
tedmd_hosvd(dimensions, downsampling_rate, integer_lag_times, threshold, max_rank, directory, res_dir)
# Run HOCUR:
tedmd_hocur(dimensions, downsampling_rate, integer_lag_times, max_rank, directory, res_dir)
print("Done")