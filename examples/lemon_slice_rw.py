import numpy as np
import matplotlib.pyplot as plt

import scikit_tt.data_driven.transform as tdt
from scikit_tt.data_driven import tgedmd
from msmtools.analysis.dense.pcca import _pcca_connected_isa

from systems import LemonSlice




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
# Define System:
LS = LemonSlice(k, beta, c=c, d=d, alpha=alpha)
# Offsets for sampling of random numbers:
offset = np.array([1.2, 1.2, 0.5, 0.5])[:, None]

""" Computational Settings: """
# Directory for results:
directory = "/Users/fkn1/Documents/Uni/Data/tgEDMD_Paper/LemonSlice/"
fig_dir = "/Users/fkn1/ICloud/Projekte/tgedmd/figures/"
# Integration time step:
dt = 1e-3
# Number of time steps:
m = 300000
m_tgedmd = 3000
delta = int(round(m / m_tgedmd))
# Number of independent simulations:
ntraj = 10
# Set ranks of SVDs:
rank_list = [50, 100, 200, 300]
# Number of timescales to record:
nits = 3

""" Definition of Gaussian basis functions """
mean_ls = np.arange(-1.2, 1.21, 0.4)
sig_ls = 0.4
mean_quad = np.arange(-1.0, 1.01, 0.5)
sig_quad = 0.5

basis_list = []
for i in range(2):
    basis_list.append([tdt.GaussFunction(i, mean_ls[j], sig_ls) for j in range(len(mean_ls))])
for i in range(2, 4):
    basis_list.append([tdt.GaussFunction(i, mean_quad[j], sig_quad) for j in range(len(mean_quad))])


""" Run Simulation """
print('Generating Data...')
data = np.zeros((ntraj, d, m))
data_tgedmd = np.zeros((ntraj, d, m_tgedmd))
rw = np.zeros((ntraj, m_tgedmd))
for ii in range(ntraj):
    for jj in range(d):
        data[ii, jj, :] = (2*offset[jj] * np.random.rand(m)) - offset[jj]
    data_tgedmd[ii, :, :] = data[ii, :, ::delta]
    rw[ii, :] = np.exp(-LS.beta * LS.potential(data_tgedmd[ii, :, :]))
print("Complete.")


""" Run tgEDMD """
timescales = np.zeros((ntraj, len(rank_list), nits))
eigfuns = np.zeros((ntraj, len(rank_list), nits+1, m_tgedmd))
for ii in range(ntraj):
    print("Analyzing data for trajectory %d..."%ii)
    diffusion = np.zeros((data_tgedmd.shape[1], data_tgedmd.shape[1], data_tgedmd.shape[2]))
    for k in range(diffusion.shape[2]):
        diffusion[:, :, k] = LS.diffusion(data_tgedmd[ii, :, k])

    # AMUSEt for the reversible case
    qq = 0
    for rank in rank_list:
        eigvals, traj_eigfuns = tgedmd.amuset_hosvd_reversible(data_tgedmd[ii, :, :], basis_list, diffusion,
                                                               reweight=rw[ii, :], num_eigvals=5,
                                                               return_option='eigenfunctionevals', threshold=0.0,
                                                               max_rank=rank)
        timescales[ii, qq, :] = [-1.0 / kappa for kappa in eigvals[1:nits+1]]
        eigfuns[ii, qq, :, :] = traj_eigfuns[:nits+1, :]
        print('Implied time scales for rank = %d : '%rank, timescales[ii, qq, :])
        qq += 1
    print(" ")

# Save Results:
dic = {}
dic["ranks"] = rank_list
dic["timescales"] = timescales
dic["eigfuns"] = eigfuns
np.savez_compressed(directory + "Results_LS_RW.npz", **dic)

""" Visualized PCCA Analysis: """
# Identify PCCA states for first trajectory and TT rank 300:
traj_eigfuns = eigfuns[0, -1, :, :]
diffs = np.abs(np.max(traj_eigfuns.T, axis=0) - np.min(traj_eigfuns.T, axis=0))
if diffs[0] > 1e-6:
    traj_eigfuns[0, :] = traj_eigfuns[0, 0] * np.ones((traj_eigfuns.shape[1]))
chi, _ = _pcca_connected_isa(traj_eigfuns.T, nits+1)
chi = chi.T
for i in range(chi.shape[1]):
    ind = np.argmax(chi[:, i])
    chi[:, i] = np.zeros((chi.shape[0],))
    chi[ind, i] = 1
chi = chi.astype(bool)

plt.figure(dpi=300)

plt.plot(data_tgedmd[0, 0, :][chi[0, :]], data_tgedmd[0, 1, :][chi[0, :]], 'bx')
plt.plot(data_tgedmd[0, 0, :][chi[1, :]], data_tgedmd[0, 1, :][chi[1, :]], 'r*')
plt.plot(data_tgedmd[0, 0, :][chi[2, :]], data_tgedmd[0, 1, :][chi[2, :]], 'g2')
plt.plot(data_tgedmd[0, 0, :][chi[3, :]], data_tgedmd[0, 1, :][chi[3, :]], 'y+')
plt.grid()
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.savefig(fig_dir + "ls_pcca_rw.pdf", bbox_inches="tight")

plt.show()