# -*- coding: utf-8 -*-

"""
Application of tensor-based EDMD (tEDMD) to molecular dynamics simulation data of the deca alanine peptide.
In this script, we use both the direct representation as well as HOCUR-based representation of the data tensor
as starting point for the method. Several choices of the maximally allowed rank are tested, and implied timescales
computed for all of these cases.

The data were generated as described in [2]. The input data consists of six files, each containing the time series of
sixteen backbone dihedral angles of the peptide. The outermost (phi-psi)-pairs were left out as they are flexible.
The files are named

DihedralTimeSeries_*.npy

Out of those sixteen angles, the ones with indices 2:12 are actually used for the calculation, resulting in a
ten-dimensional model.


References
----------
..[1] F. Nüske, P. Gelß, S. Klus, C. Clementi. "Tensor-based computation of metastable and coherent sets",
 Physica D: Nonlinear Phenomena, (2021)
..[2] F. Vitalini, A. S. J. S. Mey, F. Noé, and B. G. Keller, "Dynamic properties of force fields",
    J. Chem. Phys. (2015)
"""

import numpy as np
import scikit_tt.utils as utl
import scikit_tt.data_driven.transform as tdt
import scikit_tt.data_driven.tedmd as tedmd

import os

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

""" Basis Set: """
# Define Gaussian basis sets: we use the constant and two Gaussians for each phi-angle, and three Gaussians
# for each psi-angle:
basis_list = []
for i in range(5):
    basis_list.append([tdt.ConstantFunction(2 * i), tdt.PeriodicGaussFunction(2 * i, -2, 0.8),
                       tdt.PeriodicGaussFunction(2 * i, 1, 0.5)])
    basis_list.append([tdt.ConstantFunction(2 * i + 1), tdt.PeriodicGaussFunction(2 * i + 1, -0.5, 0.8),
                       tdt.PeriodicGaussFunction(2 * i + 1, 0, 4), tdt.PeriodicGaussFunction(2 * i + 1, 2, 0.8)])

""" Computational Settings: """
# Downsampling rate:
downsampling_rate = 10
# Define lag times in terms of physical time:
lag_times_phy = 1e-3 * np.array([100, 500, 1000, 2000, 4000, 5000, 6000])
# Build list of integer lag times:
lag_times_int = ((1000 / downsampling_rate) * lag_times_phy).astype(np.int)
# Set ranks for global SVDs and HOCUR decompositions:
rank_list = [20, 50, 100, 200, 500]
# Variable job_id contains the index of the maximal rank from rank_list to be used.
# If you're running this on a cluster, uncomment the next line to retrieve job_id from
# environment variables:
job_id = 1
#job_id = int(os.getenv("SLURM_ARRAY_TASK_ID")) - 1
max_rank = rank_list[job_id]
print("Running AMUSEt with max_rank = %d"%max_rank)


""" Load and process data: """
# Replace this line by the directory where you store the dihedral data.
directory = "/Users/fkn1/Documents/Uni/Data/Ala10_Dihedral_Data/"
#directory = "/upb/departments/pc2/users/f/fnueske/myscratch/HOSVD_ALA10_Rank_Test/"
# Replace this line by the location of your results folder:
res_dir = "/Users/fkn1/Documents/Uni/Data/21_PhysD_tEDMD_Paper/ALA10/Results/"


# Load data:
data, trajectory_lengths = load_data(directory, downsampling_rate)
# construct index for all lag times:
x_list, y_list = get_index_lists(trajectory_lengths, lag_times_int)

""" Run tEDMD using direct decomposition: """
# Apply HOSVD:
timescales_hosvd = np.zeros((len(lag_times_phy), 3))
# Compute eigenvalues and timescales:
with utl.timer() as timer:
    eigenvalues, _ = tedmd.amuset_hosvd(data, x_list, y_list, basis_list, threshold=0.0, max_rank=max_rank)
    for j in range(len(lag_times_phy)):
        timescales_hosvd[j, :] = -lag_times_phy[j] / np.log(eigenvalues[j][1:4])
cpu_time = timer.elapsed
print("Time Elapsed for AMUSEt: %.2f"%cpu_time)

# Save results to file:
dic = {}
dic["lag_times"] = lag_times_phy
dic["eigenvalues"] = eigenvalues
dic["cpu_time"] = cpu_time
#np.savez_compressed(res_dir + "Results_ALA10_HOSVD_rank_" + str(max_rank) + ".npz", **dic)

""" Run tEDMD using HOCUR decomposition: """
# Apply HOCUR:
timescales_hocur = np.zeros((len(lag_times_phy), 3))
# Compute eigenvalues and timescales:
with utl.timer() as timer:
    eigenvalues, _ = tedmd.amuset_hocur(data, x_list, y_list, basis_list, max_rank=max_rank, multiplier=2)
    for j in range(len(lag_times_phy)):
        timescales_hocur[j, :] = -lag_times_phy[j] / np.log(eigenvalues[j][1:4])
cpu_time = timer.elapsed
print("Time Elapsed for HOCUR: %.2f"%cpu_time)

# Save results to file:
dic = {}
dic["lag_times"] = lag_times_phy
dic["eigenvalues"] = eigenvalues
dic["cpu_time"] = cpu_time
#np.savez_compressed(res_dir + "Results_ALA10_HOCUR_rank_" + str(max_rank) + ".npz", **dic)
print("Done")