import numpy as np
import matplotlib.pyplot as plt

import scikit_tt.data_driven.transform as tdt
from scikit_tt.data_driven import tgedmd

from msmtools.analysis.dense.pcca import _pcca_connected_isa


""" NOTE: This script requires the system class for the Lemon Slice potential and the data. These are
            available from the authors upon request. A script to visualize the
            results and produce the figures in [1] is available on request.

    Application of tgEDMD (algorithm 2 in [1]) to simulation data of a four-dimensional toy system. The dynamics is 
    overdamped Langevin dynamics, with potential energy given by the two-dimensional "Lemon Slice"
    potential along the first two coordinates, and de-coupled harmonic potentials along the remaining
    two coordinates.
    
    The basis set consists of Gaussian functions along all coordinate directions. Several choices of
    the truncation threshold for the HOSVD of the data tensor are tested. The resulting implied time
    scales, eigenfunctions, ranks, and PCCA memberships are stored to file.
    
    References:
    [1] Lücke, M. and Nüske, F. tgEDMD: Approximation of the Kolmogorov Operator in Tensor Train Format,
        arxiv 2111.09606 (2022)
    
    
"""

"""  System Settings: """
# Number of dimensions:
d = 4
# Diffusion constant:
beta = 1.0
# Spring constant for harmonic parts of the potential
alpha = 10.0
# Pre-factor for Lemon Slice:
c = 1.0
# Number of minima for Lemon Slice:
k = 4

""" Computational Settings: """
# Location of the data:
directory = "/home/nueske/tgEDMD_Paper/Results_LemonSlice/"
# Directory for storing results:
res_dir = "/home/nueske/tgEDMD_Paper/Results_LemonSlice/"
# Directory for PCCA plots:
fig_dir = "/home/nueske/tgEDMD_Paper/Figures/"
# Integration time step:
dt = 1e-3
# Number of time steps (full data and downsampled data):
m = 300000
m_tgedmd = 3000
delta = int(round(m / m_tgedmd))
# Number of independent simulations:
ntraj = 10
# List of truncation parameters:
eps_list = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2]
# Set maximal rank:
max_rank = 1000
# Number of timescales to record:
nits = 3
# Lag times for MSM construction:
lags_msm = np.array([1, 2, 5, 10, 20, 50, 100])
# Frequency of progress messages during computation:
output_freq = 750

""" Definition of Gaussian basis functions """
# Mean values and variance of Gaussians for first two coordinates:
mean_ls = np.arange(-1.2, 1.21, 0.4)
sig_ls = 0.4
# Mean values and variance of Gaussians for last two coordinates:
mean_quad = np.arange(-1.0, 1.01, 0.5)
sig_quad = 0.5

basis_list = []
for i in range(2):
    basis_list.append([tdt.GaussFunction(i, mean_ls[j], sig_ls) for j in range(len(mean_ls))])
for i in range(2, 4):
    basis_list.append([tdt.GaussFunction(i, mean_quad[j], sig_quad) for j in range(len(mean_quad))])

""" Load the data: """
# Full data set:
data = np.load(directory + "Simulation_LS_Full.npy")
# Downsampled data set:
data_tgedmd = np.load(directory + "Simulation_LS_delta_%d.npy"%delta)

""" Run tgEDMD """
# Create arrays for timescales, eigenfunctions, TT-ranks and PCCA memberships:
timescales = np.zeros((ntraj, len(eps_list), nits))
eigfuns = np.zeros((ntraj, len(eps_list), nits+1, m_tgedmd))
ranks = np.zeros((ntraj, len(eps_list), d + 2))
chi = np.zeros((ntraj, len(eps_list), nits+1, m_tgedmd), dtype=bool)
# Run reversible tgEDMD for each trajectory independently:
for ii in range(ntraj):
    print("Analyzing data for trajectory %d..."%ii)
    # Calculate the diffusion matrix for each time step (equals a constant multiple of the identity):
    diffusion = np.repeat(np.sqrt(2.0 / beta)*np.eye(d)[:, :, None], m_tgedmd, 2)

    qq = 0
    for eps in eps_list:
        eigvals, traj_eigfuns, ranks_ii = tgedmd.amuset_hosvd(data_tgedmd[ii, :, :], basis_list, diffusion, b=None,
                                                              reweight=None, num_eigvals=5,
                                                              threshold=np.sqrt(m_tgedmd) * eps,
                                                              max_rank=max_rank, return_option='eigenfunctionevals',
                                                              output_freq=output_freq)
        # Store timescales, eigenfunctions, and ranks:
        timescales[ii, qq, :] = [-1.0 / kappa for kappa in eigvals[1:nits+1]]
        eigfuns[ii, qq, :, :] = traj_eigfuns[:nits+1, :]
        ranks[ii, qq, :] = np.array(ranks_ii)
        print('Implied time scales for eps = %.2e : ' % eps, timescales[ii, qq, :])
        # Use PCCA to compute metastable memberships:
        # Select eigenfunctions:
        traj_eigfuns = eigfuns[ii, qq, :, :]
        # Remove possible fluctuations along first eigenvector trajectory:
        diffs = np.abs(np.max(traj_eigfuns.T, axis=0) - np.min(traj_eigfuns.T, axis=0))
        if diffs[0] > 1e-6:
            traj_eigfuns[0, :] = traj_eigfuns[0, 0] * np.ones((traj_eigfuns.shape[1]))
        # Apply PCCA:
        chi_qq, _ = _pcca_connected_isa(traj_eigfuns.T, nits + 1)
        chi_qq = chi_qq.T
        # Identify state with maximal membership for each time step (hard assignment):
        for i in range(chi_qq.shape[1]):
            ind = np.argmax(chi_qq[:, i])
            chi_qq[:, i] = np.zeros((chi_qq.shape[0],))
            chi_qq[ind, i] = 1
        chi[ii, qq, :, :] = chi_qq.astype(bool)
        qq += 1
    print(" ")

# Save Results:
dic = {}
dic["eps"] = eps_list
dic["timescales"] = timescales
dic["eigfuns"] = eigfuns
dic["ranks"] = ranks
dic["chi"] = chi
np.savez_compressed(res_dir + "Results_LS_Mem.npz", **dic)

""" Create plots of PCCA states for all settings: """
for ii in range(ntraj):
    qq = 0
    for eps in eps_list:
        plt.figure(dpi=100)

        plt.plot(data_tgedmd[ii, 0, :][chi[ii, qq, 0, :]], data_tgedmd[ii, 1, :][chi[ii, qq, 0, :]], 'bx')
        plt.plot(data_tgedmd[ii, 0, :][chi[ii, qq, 1, :]], data_tgedmd[ii, 1, :][chi[ii, qq, 1, :]], 'r*')
        plt.plot(data_tgedmd[ii, 0, :][chi[ii, qq, 2, :]], data_tgedmd[ii, 1, :][chi[ii, qq, 2, :]], 'g2')
        plt.plot(data_tgedmd[ii, 0, :][chi[ii, qq, 3, :]], data_tgedmd[ii, 1, :][chi[ii, qq, 3, :]], 'y+')
        plt.grid()
        plt.xlabel(r"$x^1$")
        plt.ylabel(r"$x^2$")
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])
        plt.title("PCCA traj_id %d, eps = %.1e"%(ii, eps))
        plt.savefig(fig_dir + "PCCA_traj_id_%d_eps_%.1e.pdf"%(ii, eps), bbox_inches="tight")

        qq += 1


plt.show()
