# -*- coding: utf-8 -*-

"""
This is an example of the application of MANDy to a high-dimensional dynamical system. See [1]_ for details.

References
----------
.. [1] P. Gelß, S. Klus, J. Eisert, C. Schütte, "Multidimensional Approximation of Nonlinear Dynamical Systems",
       arXiv:1809.02448, 2018
"""

import numpy as np
import scipy.linalg as splin
from scikit_tt.tensor_train import TT
import scikit_tt.mandy as mandy
import scikit_tt.models as mdl
import scikit_tt.utils as utl
import matplotlib.pyplot as plt
#import matplotlib.lines as lin
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm

utl.header('MANDy - Fermi-Pasta-Ulam problem')

# model parameters
d = 10
psi = [lambda x: 1, lambda x: x, lambda x: x ** 2, lambda x: x ** 3]
p = len(psi)

# construct exact solution in TT and matrix format
utl.progress('Construct exact solution in TT format', 0, dots=7)
xi_exact = mdl.fermi_pasta_ulam_solution(d)
utl.progress('Construct exact solution in TT format', 100, dots=7)
utl.progress('Construct exact solution in matrix format', 0)
xi_exact_mat = xi_exact.full().reshape([p ** d, d])
utl.progress('Construct exact solution in matrix format', 100)

# loop parameters
snapshots_min = 100
snapshots_max = 500
snapshots_step = 100
snapshots_mat = 5000

# define arrays for CPU times and relative errors
cpu_times = np.zeros([6, int((snapshots_max - snapshots_min)/snapshots_step) + 1])
rel_errors = np.zeros([6, int((snapshots_max - snapshots_min)/snapshots_step) + 1])

# compare CPU times of tensor-based and matrix-based approaches
for i in range(snapshots_min, snapshots_max + snapshots_step, snapshots_step):

    print('\nNumber of snapshots: ' + str(i))
    print('-' * (21 + len(str(i))) + '\n')

    # storing index
    ind = int((i-snapshots_min)/snapshots_step)

    # generate data
    utl.progress('Generate test data', 0, dots=13)
    [x, y] = mdl.fermi_pasta_ulam_data(d, i)
    utl.progress('Generate test data', 100, dots=13)

    # computation in matrix format
    if i < snapshots_mat:
        utl.progress('Running matrix-based MANDy', 0, dots=5)

        # construct psi_x in matrix format
        psi_x = np.zeros([p ** d, i])
        for j in range(i):
            c = [psi[l](x[0, j]) for l in range(p)]
            for k in range(1, d):
                # c = np.kron(c,[psi[l](x[k, j]) for l in range(p)])
                c = np.tensordot(c,[psi[l](x[k, j]) for l in range(p)],axes=0)
            psi_x[:, j] = c.reshape(psi_x.shape[0])

        utl.progress('Running matrix-based MANDy', 50, dots=5)

        # compute xi in matrix format
        with utl.Timer() as time:
            [u, s, v] = splin.svd(psi_x, full_matrices=False, overwrite_a=True, check_finite=False, lapack_driver='gesvd')
            xi = y @ v.transpose() @ np.diag(np.reciprocal(s)) @ u.transpose()
            # xi = y @ np.linalg.pinv(psi_x)
        cpu_times[0, ind] = time.elapsed
        rel_errors[0, ind] = np.linalg.norm(xi.transpose() - xi_exact_mat) / np.linalg.norm(xi_exact_mat)
        del xi, u, s, v
        utl.progress('Running matrix-based MANDy', 100, dots=5)
        print('   CPU time      : ' + str("%.2f" % cpu_times[0, ind]) + 's')
        print('   relative error: ' + str("%.2e" % rel_errors[0, ind]))

    else:
        cpu_times[0, ind] = 2 * cpu_times[0, ind - 1] - cpu_times[0, ind - 2]

    # exact computation in TT format
    utl.progress('Running MANDy (eps=0)', 0, dots=10)
    with utl.Timer() as time:
        xi = mandy.mandy_cm(x, y, psi, threshold=0)
    cpu_times[1, ind] = time.elapsed
    rel_errors[1, ind] = (xi - xi_exact).norm() / xi_exact.norm()
    del xi
    utl.progress('Running MANDy (eps=0)', 100, dots=10)
    print('   CPU time      : ' + str("%.2f" % cpu_times[1, ind]) + 's')
    print('   relative error: ' + str("%.2e" % rel_errors[1, ind]))

    # use thresholds larger 0 for orthonormalizations
    for j in range(0, 4):
        utl.progress('Running MANDy (eps=10^' + str(-10+j) + ')', 0, dots=8-len(str(-10+j)))
        with utl.Timer() as time:
            xi = mandy.mandy_cm(x, y, psi, threshold=10 ** (-10 + j))
        cpu_times[j + 2, ind] = time.elapsed
        rel_errors[j + 2, ind] = (xi - xi_exact).norm() / xi_exact.norm()
        del xi
        utl.progress('Running MANDy (eps=10^' + str(-10+j) + ')', 100, dots=8-len(str(-10+j)))
        print('   CPU time      : ' + str("%.2f" % cpu_times[j + 2, ind]) + 's')
        print('   relative error: ' + str("%.2e" % rel_errors[j + 2, ind]))

# plot results
# ------------

utl.plot_parameters()

# CPU times
plt.figure(dpi=300)
x_values = np.arange(snapshots_min, snapshots_max + 1, snapshots_step)
plt.plot(x_values[:-2], cpu_times[0, :-2], label=r"Matrix repr.")
plt.plot(x_values[-3:], cpu_times[0, -3:], 'C0--')
plt.plot(x_values, cpu_times[1, :], label=r"TT - exact")
plt.plot(x_values, cpu_times[2, :], label=r"TT - $\varepsilon=10^{-10}$")
plt.plot(x_values, cpu_times[3, :], label=r"TT - $\varepsilon=10^{-9}$")
plt.plot(x_values, cpu_times[4, :], label=r"TT - $\varepsilon=10^{-8}$")
plt.plot(x_values, cpu_times[5, :], label=r"TT - $\varepsilon=10^{-7}$")
plt.gca().set_xlim([snapshots_min, snapshots_max])
plt.xticks(np.arange(snapshots_min, snapshots_max + 1, 2 * snapshots_step))
# plt.gca().set_ylim([0, 1500])
# plt.yticks(np.arange(0, 1750, 250))
plt.title(r'CPU times for $d = 10$', y=1.03)
plt.xlabel(r'$m$')
plt.ylabel(r'$T / s$')
plt.legend(loc=2, fontsize=12).get_frame().set_alpha(1)
plt.show()

# relative errors
plt.figure(dpi=300)
x_values = np.arange(snapshots_min, snapshots_max + 1, snapshots_step)
plt.semilogy(x_values, rel_errors[0,:],label="Matrix - exact")
for j in range(5):
    exp = -11 + j
    if exp == -11:
        plt.semilogy(x_values, rel_errors[1,:],label=r'TT - exact')
    else:
        plt.semilogy(x_values, rel_errors[1+j,:], label=r'TT - $\varepsilon = 10^{' + str(exp) + '}$')
plt.gca().set_ylim([10**-4,10**-1])
plt.grid(which='major')
plt.grid(which='minor')
plt.xticks(np.arange(snapshots_min, snapshots_max + 1, 2 * snapshots_step))
plt.yticks([10**-4, 10**-3, 10**-2, 10**-1])
plt.title(r'Relative errors for $d = 10$', y=1.03)
plt.xlabel(r'$m$')
plt.ylabel(r'$|| \Xi_{\textrm{exact}} - \Xi ||~/~|| \Xi_{\textrm{exact}} ||$')
plt.gca().set_xlim([snapshots_min, snapshots_max - 2 * snapshots_step])
plt.legend(loc=1, fontsize=12).get_frame().set_alpha(1)
plt.show()


# relative errors d=10

# relative errors for eps=10^-10

# Xi_exact_tt = construct_Xi_exact(10)
# Xi_exact_mat = Xi_exact_tt.full().reshape(np.prod(Xi_exact_tt.row_dims[:-1]), Xi_exact_tt.row_dims[-1], order='F')
#
#
# m_start = 1000
# m_step = 500
# m_number_mat = 9
# m_number_tt = 11
# exp_start = -11
# exp_step = 1
# exp_number = 5
# repeats = 1
#
# # computations
#
# m_mat = [0] * m_number_mat
# T_mat = [0] * m_number_mat
# E_mat = [0] * m_number_mat
# for i in range(m_number_mat):
#     m_tmp = m_start + i * m_step
#     print("Running matrix-based MANDy for m="+str(m_tmp))
#     for k in range(repeats):
#         Xi_mat, time = mandy.mandy_basis_major_matrix(X[:,:m_tmp], Y[:,:m_tmp], psi, cpu_time=True)
#         T_mat[i] = T_mat[i]+time.elapsed
#     m_mat[i] = m_tmp
#     T_mat[i] = T_mat[i]/repeats
#
#     E_mat[i] = np.linalg.norm(Xi_mat.transpose() - Xi_exact_mat)/np.linalg.norm(Xi_exact_mat)
# T_tt = np.zeros([exp_number, m_number_tt])
# E_tt = np.zeros([exp_number, m_number_tt])
# for j in range(exp_number):
#     m_tt = [0] * m_number_tt
#     for i in range(m_number_tt):
#         m_tmp = m_start + i * m_step
#         exp = exp_start + j * exp_step
#         if exp == -11:
#             print("Running MANDy for m="+str(m_tmp)+" and eps=0")
#         else:
#             print("Running MANDy for m=" + str(m_tmp) + " and eps=10^" + str(exp))
#         for k in range(repeats):
#             if exp == -11:
#                 Xi_tt, time = mandy.mandy_basis_major(X[:,:m_tmp], Y[:,:m_tmp], psi, 0, cpu_time=True)
#             else:
#                 Xi_tt, time = mandy.mandy_basis_major(X[:, :m_tmp], Y[:, :m_tmp], psi, threshold=10 ** exp, cpu_time=True)
#             T_tt[j,i] = T_tt[j,i]+time.elapsed
#         m_tt[i] = m_tmp
#         T_tt[j,i] = T_tt[j,i]/repeats
#         # Xi_tt = Xi_tt.full().reshape(np.prod(Xi_tt.row_dims[:-1]), Xi_tt.row_dims[-1], order='F')
#         E_tt[j,i] = scikit_tt.norm(Xi_tt - Xi_exact_tt) / np.linalg.norm(Xi_exact_tt)
# np.savez('data/example_mandy_fpu_test', m_mat=m_mat, T_mat=T_mat, E_mat=E_mat, m_tt=m_tt, T_tt=T_tt, E_tt=E_tt)
#
#
# plots

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.rcParams["mathtext.fontset"] = "cm"
# plt.rcParams.update({'font.size': 18})
# plt.rcParams.update({'figure.autolayout': True})
#
# npzfile = np.load('data/example_mandy_fpu_1.npz') # load data
# m_mat = npzfile['m_mat']
# T_mat = npzfile['T_mat']
# E_mat = npzfile['E_mat']
# m_tt = npzfile['m_tt']
# T_tt = npzfile['T_tt']
# E_tt = npzfile['E_tt']
#
# m_int = np.vstack([np.square(m_mat),np.array(m_mat),np.ones([1,len(m_mat)])]) # extrapolate CPU times
# T_int = T_mat.copy()
# m_int[:,-1] = 100*m_int[:,-1]
# T_int[-1] = 100*T_int[-1]
#
# p = np.linalg.lstsq(m_int.transpose(),T_int.transpose(), rcond=None)
# m_int = np.vstack([[np.square(m_tt)], [np.array(m_tt)], [np.ones([len(m_tt)])]])
# T_int = p[0].transpose() @ m_int
#
# plt.figure(dpi=300)
# plt.plot(m_mat, T_mat,label="Matrix repr.")
# plt.plot(m_tt[-(len(m_tt)-len(m_mat)+1):], T_int[-(len(m_tt)-len(m_mat)+1):],'C0--')
# for j in range(exp_number):
#     exp = exp_start + j * exp_step
#     if exp == -11:
#         plt.plot(m_tt, T_tt[j, :], label=r'TT - exact')
#     else:
#         plt.plot(m_tt, T_tt[j,:],label=r'TT - $\varepsilon = 10^{'+str(exp)+'}$')
# plt.gca().set_xlim([1000,6000])
# plt.gca().set_ylim([0,250])
# plt.grid()
# plt.xticks(np.arange(1000,7000,1000))
# plt.yticks(np.arange(0,1750,250))
# plt.title(r'CPU times for $d = 10$', y=1.03)
# plt.xlabel(r'$m$')
# plt.ylabel(r'$T / s$')
# plt.legend(loc=2, fontsize=12).get_frame().set_alpha(1)
# plt.savefig('example_fpu_1_a.pdf', dpi='figure', format='pdf', bbox_inches='tight')
# plt.show()
#
# plt.figure(dpi=300)
# plt.semilogy(m_mat, E_mat,label="Matrix - exact")
# for j in range(exp_number):
#     exp = exp_start + j * exp_step
#     if exp == -11:
#         plt.semilogy(m_mat, E_tt[j,:len(m_mat)],label=r'TT - exact')
#     else:
#         plt.semilogy(m_mat, E_tt[j, :len(m_mat)], label=r'TT - $\varepsilon = 10^{' + str(exp) + '}$')
# plt.gca().set_xlim([1000,5000])
# plt.gca().set_ylim([10**-4,10**-1])
# plt.grid(which='major')
# plt.grid(which='minor')
# plt.xticks(np.arange(1000,6000,1000))
# plt.yticks([10**-4, 10**-3, 10**-2, 10**-1])
# plt.title(r'Relative errors for $d = 10$', y=1.03)
# plt.xlabel(r'$m$')
# plt.ylabel(r'$|| \Xi_{\textrm{exact}} - \Xi ||~/~|| \Xi_{\textrm{exact}} ||$')
# plt.legend(loc=1, fontsize=12).get_frame().set_alpha(1)
# plt.savefig('example_fpu_1_b.pdf', dpi='figure', format='pdf', bbox_inches='tight')
# plt.show()


# second experiment


#
#
#
#
#
#
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.rcParams["mathtext.fontset"] = "cm"
# plt.rcParams.update({'font.size': 19})
# plt.rcParams.update({'figure.autolayout': True})
#
#
#
#
# psi = [lambda x: 1, lambda x: x, lambda x: x ** 2, lambda x: x ** 3]  # basis functions
#
#
#
# m_start = 500
# m_step = 500
# m_number = 12
# d_start = 2
# d_step = 1
# d_number = 19
#
# # computations
#
# # E = np.zeros([d_number, m_number])
# # for j in range(d_number):
# #     for i in range(m_number):
# #         m = m_start + i * m_step
# #         d = d_start + j * d_step
# #         print("Running MANDy for d=" + str(d) + ", m=" + str(m) + ", and eps=0")
# #         D = scipy.io.loadmat("data/FPU_d"+str(d)+"_m6000.mat")  # load data
# #         X = D["X"]
# #         Y = D["Y"]
# #         Xi_tt = mandy.mandy_basis_major(X[:, :m], Y[:, :m], psi, threshold=10**-10)
# #         # Xi_tt = Xi_tt.full().reshape(np.prod(Xi_tt.row_dims[:-1]), Xi_tt.row_dims[-1], order='F')
# #         Xi = construct_Xi_exact(d)
# #         Xi_tmp = Xi_tt - Xi
# #         E[j,i] = Xi_tmp.norm()/Xi.norm()
# #         # E[j,i] = scikit_tt.norm(Xi_tt - Xi) / np.linalg.norm(Xi)
# #         print(E[j,i])
# # np.savez('data/example_mandy_fpu_b', E=E)
#
#
# # plots
#
# npzfile = np.load('data/example_mandy_fpu_b.npz') # load data
# E = npzfile['E']
#
#
# plt.figure(dpi=300, figsize=(7.2,5.4))
# # im = plt.imshow(E.T[::-1,:])
# # plt.colorbar(im, fraction=0.046, pad=0.04)
# ax = plt.gca()
# im = ax.imshow(np.log10(E.T[::-1,1:]),cmap='jet')
# plt.plot([8.5,8.5,7.5,7.5,6.5,6.5],[11.5,9.5,9.5,1.5,1.5,-0.5],'--', color='black', linewidth = 2)
#
#
# plt.xticks(np.arange(1,18,2),np.arange(4,22,2))
# plt.yticks(np.arange(0,12,2),np.arange(6000,0,-1000))
# plt.title(r'Relative errors for $\varepsilon = 10^{-10}$', y=1.03)
# plt.xlabel(r'$d$')
# plt.ylabel(r'$m$')
# # create an axes on the right side of ax. The width of cax will be 5%
# # of ax and the padding between cax and ax will be fixed at 0.05 inch.
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.15)
# cbar =  plt.colorbar(im, cax=cax)
# cbar.ax.tick_params(labelsize=18)
# cbar.ax.set_yticklabels([r'$10^{-7}$',r'$10^{-6}$',r'$10^{-5}$',r'$10^{-4}$',r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$'])
#
# # plt.gca().set_xlim([1000,6000])
# # plt.gca().set_ylim([0,250])
#
# # plt.yticks(np.arange(0,1750,250))
# plt.savefig('example_fpu_2.pdf', dpi='figure', format='pdf', bbox_inches='tight')
# plt.show()
#
#
