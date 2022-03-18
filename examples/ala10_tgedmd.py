# -*- coding: utf-8 -*-

"""
    NOTE: This script requires the ala10 trajectory data, which is not included in the scikit-tt package.
        Configure the variable "data_path" to point to the directory where the data is stored on your machine.

    Application of tgEDMD (algorithm 2 in [1]) to trajectory data of the ten residue peptide deca-alanine, as described
    in [1, section 5.2].
    The basis set consists of Gaussian functions along all coordinate directions.
    Several choices of the truncation threshold for the HOSVD of the data tensor and downsampling ratios are tested.
    The resulting implied time scales, ranks, and PCCA memberships are plotted.

    References
    ----------
    [1] M. Lücke and F. Nüske, "tgEDMD: Approximation of the Kolmogorov Operator in Tensor Train Format",
        arXiv:2111.09606, 2022
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from msmtools.analysis.dense.pcca import _pcca_connected_isa
import scikit_tt.utils as utl
import scikit_tt.data_driven.transform as tdt
import scikit_tt.data_driven.tgedmd as tgedmd


def main():
    # define parameters
    data_path = "data/Ala10/"
    num_snapshots = [300, 3000]  # trajectory is downsampled to num_snapshots
    eps_exps = [4, 6]  # epsilon = 10^-{eps_exp}; relative SVD truncation threshold = sqrt(num_snapshots) * epsilon
    rank_cap = 500  # the TT ranks are capped at rank_cap

    # load trajectory data
    traj = [np.load(data_path + f"Dih_Traj_{i}.npy") for i in range(6)]
    traj = np.concatenate(traj, axis=0).T
    traj = traj[2:12, :]  # we use the 10 internal dihedral angles
    d, m = traj.shape

    traj_sigma = [np.load(data_path + f"Dih_Jac_Traj_{i}.npy") for i in range(6)]
    traj_sigma = [t[2:12, :, :, :] for t in traj_sigma]  # we use the 10 internal dihedral angles
    traj_sigma = np.concatenate(traj_sigma, axis=3)
    traj_sigma = np.reshape(traj_sigma, (d, traj_sigma.shape[1] * traj_sigma.shape[2], -1))
    # traj_sigma[i, :, j] are the partial derivatives of the i-th dihedral angle w.r.t. the atomic positions
    # at the j-th snapshot

    # define basis functions, for details see [1, section 5.2]
    basis_list = []
    for i in range(5):
        basis_list.append([tdt.ConstantFunction(2 * i), tdt.PeriodicGaussFunction(2 * i, -2, 0.8),
                           tdt.PeriodicGaussFunction(2 * i, 1, 0.5)])
        basis_list.append([tdt.ConstantFunction(2 * i + 1), tdt.PeriodicGaussFunction(2 * i + 1, -0.5, 0.8),
                           tdt.PeriodicGaussFunction(2 * i + 1, 0, 4), tdt.PeriodicGaussFunction(2 * i + 1, 2, 0.8)])

    # run tgEDMD for the different parameters specified above (eps_exps and num_snapshots)
    timescales = []
    ranks = []
    for num_snap in num_snapshots:
        # downsample the original data to the desired num_snapshots
        idx_snapshots = np.unique(np.linspace(0, m - 1, num_snap).astype(int))
        this_traj = traj[:, idx_snapshots]
        this_traj_sigma = traj_sigma[:, :, idx_snapshots]

        for eps_exp in eps_exps:
            print(f"Running tgEDMD with epsilon = 10^-{eps_exp} and {num_snap} snapshots.")
            threshold = (num_snap ** 0.5) * 10 ** (-eps_exp)

            with utl.timer() as timer:
                eigenvalues, traj_eigfuns, rks = tgedmd.amuset_hosvd(this_traj, basis_list, this_traj_sigma,
                                                                     num_eigvals=5, threshold=threshold,
                                                                     max_rank=rank_cap,
                                                                     output_freq=int(this_traj.shape[1] / 10),
                                                                     rel_threshold=True)
            cpu_time = timer.elapsed
            print(f"Time Elapsed: {cpu_time}")

            # store the timescales and ranks for plotting later
            timescales.append(np.array([-1 / kappa for kappa in eigenvalues[1:]]))
            ranks.append(rks)

            print("--------------------------")
            print("")

    print("Plotting results ...")
    plot_timescales(eps_exps, num_snapshots, data_path, timescales)
    plot_ranks(eps_exps, num_snapshots, ranks)

    chi = calc_chi(traj_eigfuns, 2)  # array that assigns each snapshot to one of two clusters
    plot_density(eps_exps[-1], num_snapshots[-1], this_traj, chi)

    plt.show()


def plot_timescales(eps_exp, num_snapshots, data_path, timescales):
    its_ref = np.load(data_path + "Timescales_MSM.npy")[5, :3] / 1000

    plt.figure(figsize=(5, 4))
    plt.rc('text', usetex=True)
    plt.rc('font', family='sans')
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'figure.autolayout': True})
    plt.rcParams.update({'axes.grid': True})

    colors = ["tab:blue", "tab:orange", "tab:green"]
    styles = ["o--", "x-."]
    handles = []

    msm = plt.plot([1, 2, 3], its_ref, 'kx-', label="MSM")[0]

    for idx_m, m in enumerate(num_snapshots):
        for idx_e, expon in enumerate(eps_exp):
            its = timescales.pop(0)
            its = its[:3]
            factors = its_ref[0] / its[0]
            its = (its.T * factors).T

            handles.append(plt.plot([1, 2, 3], its, styles[idx_m], c=colors[idx_e], label=f"1.0e-{expon}, m={m}")[0])

        if idx_m < len(num_snapshots) - 1:  # for legend formatting
            handles.append(plt.plot([], [], color=(0, 0, 0, 0), label=" ")[0])
    handles.append(msm)

    plt.xlim((0.95, 3.3))
    plt.legend(handles=handles, ncol=2, fontsize=11)
    plt.xticks([1, 2, 3])
    plt.ylabel(r"$t_i$ [ns]")
    plt.xlabel("i")
    plt.title("Implied Time Scales")
    # plt.savefig("timescales.pdf", bbox_inches="tight")


def plot_ranks(eps_exp, num_snapshots, ranks):
    colors = ["tab:blue", "tab:orange", "tab:green"]
    styles = ["o--", "x-."]

    plt.rc('text', usetex=True)
    plt.rc('font', family='sans')
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'figure.autolayout': True})
    plt.rcParams.update({'axes.grid': True})
    fig, ax = plt.subplots(figsize=(5, 4))

    for idx_m, m in enumerate(num_snapshots):
        for idx_e, expon in enumerate(eps_exp):
            rks = ranks.pop(0)[:-1]
            plt.plot(np.arange(len(rks)), rks, styles[idx_m], c=colors[idx_e], label=f"1.0e-{expon}, m={m}")

    ax.set_yscale('log')
    plt.legend(ncol=2, fontsize=11)
    plt.xlim((-0.5, 11.5))
    plt.ylim((0.3, 600))
    plt.xlabel("k")
    plt.ylabel("$r_k$")
    plt.title("TT-ranks Data Tensor")
    # plt.savefig("ranks.pdf", bbox_inches="tight")


def calc_chi(traj_eigfuns, num_clusters):
    """
    Assign snapshots to clusters.
    """
    k = num_clusters
    traj_eigfuns[0, :] = traj_eigfuns[0, 0] * np.ones(traj_eigfuns.shape[1])
    chi, _ = _pcca_connected_isa(traj_eigfuns.T, k)
    chi = chi.T
    for i in range(chi.shape[1]):
        ind = np.argmax(chi[:, i])
        chi[:, i] = np.zeros((chi.shape[0],))
        chi[ind, i] = 1
    chi = chi.astype(bool)
    return chi


def plot_alpha_beta_gamma(ax):
    lw = 0.8
    ax.plot([-np.pi, 0], [-2, -2], 'k', linewidth=lw)
    ax.plot([-np.pi, 0], [1.4, 1.4], 'k', linewidth=lw)
    ax.plot([2.3, np.pi], [-2, -2], 'k', linewidth=lw)
    ax.plot([2.3, np.pi], [1.4, 1.4], 'k', linewidth=lw)
    ax.axvline(0, color='k', linewidth=lw)
    ax.axvline(2.3, color='k', linewidth=lw)


def plot_density(eps_exp, num_snaps, traj, chi):
    heatmaps0 = []
    heatmaps1 = []
    extents0 = []
    extents1 = []

    for i in range(5):
        x = traj[2 * i, :][chi[0, :]]
        y = traj[2 * i + 1, :][chi[0, :]]
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=50, normed=True, range=[[-np.pi, np.pi], [-np.pi, np.pi]])
        heatmaps0.append(heatmap)
        extents0.append([xedges[0], xedges[-1], yedges[0], yedges[-1]])

        x = traj[2 * i, :][chi[1, :]]
        y = traj[2 * i + 1, :][chi[1, :]]
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=50, normed=True, range=[[-np.pi, np.pi], [-np.pi, np.pi]])
        heatmaps1.append(heatmap)
        extents1.append([xedges[0], xedges[-1], yedges[0], yedges[-1]])

    vmax0 = max([np.max(heatmap) for heatmap in heatmaps0])
    vmax1 = max([np.max(heatmap) for heatmap in heatmaps1])

    fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(10, 4), gridspec_kw={'width_ratios': [10, 10, 10, 10, 10, 1]})
    plt.rc('text', usetex=True)
    plt.rc('font', family='sans')
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams.update({'font.size': 11})
    plt.rcParams.update({'figure.autolayout': True})
    plt.rcParams.update({'axes.grid': True})
    for i in range(5):

        # plot density
        mappable0 = axes[1][i].imshow(heatmaps0[i].T, extent=extents0[i], origin='lower',
                                     cmap=cm.Reds, vmin=0, vmax=vmax0)
        mappable1 = axes[0][i].imshow(heatmaps1[i].T, extent=extents1[i], origin='lower',
                                      cmap=cm.Blues, vmin=0, vmax=vmax1)

        plot_alpha_beta_gamma(axes[0][i])
        plot_alpha_beta_gamma(axes[1][i])
        axes[0][i].set_ylim((-np.pi, np.pi))
        axes[0][i].set_xlim((-np.pi, np.pi))
        axes[1][i].set_ylim((-np.pi, np.pi))
        axes[1][i].set_xlim((-np.pi, np.pi))

        axes[0][i].set_yticks([])
        axes[0][i].set_xticks([])
        axes[1][i].set_yticks([])
        axes[1][i].set_xticks([])

    fig.colorbar(mappable0, cax=axes[1][5])
    fig.colorbar(mappable1, cax=axes[0][5])
    fig.suptitle(f"Metastable Decomposition, epsilon=1e-{eps_exp}, {num_snaps} snapshots", fontsize=16)
    # plt.savefig("pcca.pdf", bbox_inches="tight")


if __name__ == '__main__':
    main()
