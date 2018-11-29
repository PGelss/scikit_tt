#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy.io
import mandy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lin
import scikit_tt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm



def construct_Xi_exact(d):
    """Construction of the exact solution of the FPU example in TT format.

    Reference: ...

    Arguments
    ---------
    d: int
        number of oscillators

    Returns
    -------
    Xi_exact: tensor train
        exact coefficient tensor
    """
    core_type_1 = np.zeros([1,4,1,1]) # define core types
    core_type_1[0,0,0,0] = 1
    core_type_2 = np.eye(4).reshape([1,4,1,4])
    core_type_3 = np.zeros([4,4,1,4])
    core_type_3[0, 1, 0, 0] = -2
    core_type_3[0, 3, 0, 0] = -1.4
    core_type_3[0, 0, 0, 1] = 1
    core_type_3[0, 2, 0, 1] = 2.1
    core_type_3[0, 1, 0, 2] = -2.1
    core_type_3[0, 0, 0, 3] = 0.7
    core_type_3[1, 0, 0, 0] = 1
    core_type_3[1, 2, 0, 0] = 2.1
    core_type_3[2, 1, 0, 0] = -2.1
    core_type_3[3, 0, 0, 0] = 0.7
    core_type_4 = np.eye(4).reshape([4,4,1,1])
    cores = [None] * (d+1)
    cores[0] = np.zeros([1,4,1,4])
    cores[0][0,:,:,:] = core_type_3[0,:,:,:]
    cores[1] = core_type_4
    for i in range(2,d):
        cores[i] = core_type_1
    cores[d] = np.zeros([1,d,1,1])
    cores[d][0,0,0,0] = 1
    Xi_exact = scikit_tt.TT(cores)
    for k in range(1,d-1):
        cores = [None] * (d + 1)
        for i in range(k-1):
            cores[i] = core_type_1
        cores[k-1] = core_type_2
        cores[k] = core_type_3
        cores[k+1] = core_type_4
        for i in range(k+2,d):
            cores[i] = core_type_1
        cores[d] = np.zeros([1,d,1,1])
        cores[d][0,k,0,0] = 1
        Xi_exact = Xi_exact + scikit_tt.TT(cores)
    cores = [None] * (d + 1)
    for i in range(d-2):
        cores[i] = core_type_1
    cores[d-2] = core_type_2
    cores[d-1] = np.zeros([4,4,1,1])
    cores[d-1][:,:,:,0] = core_type_3[:,:,:,0]
    cores[d] = np.zeros([1, d, 1, 1])
    cores[d][0, d-1, 0, 0] = 1
    Xi_exact = Xi_exact + scikit_tt.TT(cores)
    return Xi_exact


# D = scipy.io.loadmat("data/FPU_d10_m6000.mat")  # load data
# X = D["X"]
# Y = D["Y"]
# psi = [lambda x: 1, lambda x: x, lambda x: x ** 2, lambda x: x ** 3]  # basis functions
#
# Xi_exact_tt = construct_Xi_exact(10)
# Xi_exact_mat = Xi_exact_tt.full().reshape(np.prod(Xi_exact_tt.row_dims[:-1]), Xi_exact_tt.row_dims[-1], order='F')
#
#
m_start = 1000
m_step = 500
m_number_mat = 9
m_number_tt = 11
exp_start = -11
exp_step = 1
exp_number = 5
repeats = 1
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








plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams.update({'font.size': 19})
plt.rcParams.update({'figure.autolayout': True})




psi = [lambda x: 1, lambda x: x, lambda x: x ** 2, lambda x: x ** 3]  # basis functions



m_start = 500
m_step = 500
m_number = 12
d_start = 2
d_step = 1
d_number = 19

# computations

# E = np.zeros([d_number, m_number])
# for j in range(d_number):
#     for i in range(m_number):
#         m = m_start + i * m_step
#         d = d_start + j * d_step
#         print("Running MANDy for d=" + str(d) + ", m=" + str(m) + ", and eps=0")
#         D = scipy.io.loadmat("data/FPU_d"+str(d)+"_m6000.mat")  # load data
#         X = D["X"]
#         Y = D["Y"]
#         Xi_tt = mandy.mandy_basis_major(X[:, :m], Y[:, :m], psi, threshold=10**-10)
#         # Xi_tt = Xi_tt.full().reshape(np.prod(Xi_tt.row_dims[:-1]), Xi_tt.row_dims[-1], order='F')
#         Xi = construct_Xi_exact(d)
#         Xi_tmp = Xi_tt - Xi
#         E[j,i] = Xi_tmp.norm()/Xi.norm()
#         # E[j,i] = scikit_tt.norm(Xi_tt - Xi) / np.linalg.norm(Xi)
#         print(E[j,i])
# np.savez('data/example_mandy_fpu_b', E=E)


# plots

npzfile = np.load('data/example_mandy_fpu_b.npz') # load data
E = npzfile['E']


plt.figure(dpi=300, figsize=(7.2,5.4))
# im = plt.imshow(E.T[::-1,:])
# plt.colorbar(im, fraction=0.046, pad=0.04)
ax = plt.gca()
im = ax.imshow(np.log10(E.T[::-1,1:]),cmap='jet')
plt.plot([8.5,8.5,7.5,7.5,6.5,6.5],[11.5,9.5,9.5,1.5,1.5,-0.5],'--', color='black', linewidth = 2)


plt.xticks(np.arange(1,18,2),np.arange(4,22,2))
plt.yticks(np.arange(0,12,2),np.arange(6000,0,-1000))
plt.title(r'Relative errors for $\varepsilon = 10^{-10}$', y=1.03)
plt.xlabel(r'$d$')
plt.ylabel(r'$m$')
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.15)
cbar =  plt.colorbar(im, cax=cax)
cbar.ax.tick_params(labelsize=18)
cbar.ax.set_yticklabels([r'$10^{-7}$',r'$10^{-6}$',r'$10^{-5}$',r'$10^{-4}$',r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$'])

# plt.gca().set_xlim([1000,6000])
# plt.gca().set_ylim([0,250])

# plt.yticks(np.arange(0,1750,250))
plt.savefig('example_fpu_2.pdf', dpi='figure', format='pdf', bbox_inches='tight')
plt.show()


