#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy.io

from scikit_tt import mandy

import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'figure.autolayout': True})


D = scipy.io.loadmat("data/FPU_d10_m6000.mat")  # load data
X = D["X"]
Y = D["Y"]
Xi = D["Xi"]
psi = [lambda x: 1, lambda x: x, lambda x: x ** 2, lambda x: x ** 3]  # basis functions

m_start = 1000
m_step = 500
m_number_mat = 9
m_number_tt = 11
exp_start = -11
exp_step = 1
exp_number = 5
repeats = 3

# computations

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
#     E_mat[i] = np.linalg.norm(Xi_mat.transpose() - Xi)/np.linalg.norm(Xi)
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
#         Xi_tt = Xi_tt.full().reshape(np.prod(Xi_tt.row_dims[:-1]), Xi_tt.row_dims[-1], order='F')
#         E_tt[j,i] = np.linalg.norm(Xi_tt - Xi) / np.linalg.norm(Xi)
# np.savez('data/example_mandy_fpu_1', m_mat=m_mat, T_mat=T_mat, E_mat=E_mat, m_tt=m_tt, T_tt=T_tt, E_tt=E_tt)


# plots

npzfile = np.load('data/example_mandy_fpu_1.npz') # load data
m_mat = npzfile['m_mat']
T_mat = npzfile['T_mat']
E_mat = npzfile['E_mat']
m_tt = npzfile['m_tt']
T_tt = npzfile['T_tt']
E_tt = npzfile['E_tt']

m_int = np.vstack([np.square(m_mat),np.array(m_mat),np.ones([1,len(m_mat)])]) # extrapolate CPU times
T_int = T_mat.copy()
m_int[:,-1] = 100*m_int[:,-1]
T_int[-1] = 100*T_int[-1]

p = np.linalg.lstsq(m_int.transpose(),T_int.transpose(), rcond=None)
m_int = np.vstack([[np.square(m_tt)], [np.array(m_tt)], [np.ones([len(m_tt)])]])
T_int = p[0].transpose() @ m_int

plt.figure(dpi=300)
plt.plot(m_mat, T_mat,label="Matrix repr.")
plt.plot(m_tt[-(len(m_tt)-len(m_mat)+1):], T_int[-(len(m_tt)-len(m_mat)+1):],'C0--')
for j in range(exp_number):
    exp = exp_start + j * exp_step
    if exp == -11:
        plt.plot(m_tt, T_tt[j, :], label=r'TT - exact')
    else:
        plt.plot(m_tt, T_tt[j,:],label=r'TT - $\varepsilon = 10^{'+str(exp)+'}$')
plt.gca().set_xlim([1000,6000])
plt.gca().set_ylim([0,250])
plt.grid()
plt.xticks(np.arange(1000,7000,1000))
plt.yticks(np.arange(0,1750,250))
plt.title(r'CPU times for $d = 10$', y=1.03)
plt.xlabel(r'$m$')
plt.ylabel(r'$T / s$')
plt.legend(loc=2, fontsize=12).get_frame().set_alpha(1)
plt.savefig('example_fpu_1_a.pdf', dpi='figure', format='pdf')
plt.show()

plt.figure(dpi=300)
plt.semilogy(m_mat, E_mat,label="Matrix - exact")
for j in range(exp_number):
    exp = exp_start + j * exp_step
    if exp == -11:
        plt.semilogy(m_mat, E_tt[j,:len(m_mat)],label=r'TT - exact')
    else:
        plt.semilogy(m_mat, E_tt[j, :len(m_mat)], label=r'TT - $\varepsilon = 10^{' + str(exp) + '}$')
plt.gca().set_xlim([1000,5000])
plt.gca().set_ylim([10**-4,10**-1])
plt.grid(which='major')
plt.grid(which='minor')
plt.xticks(np.arange(1000,6000,1000))
plt.yticks([10**-4, 10**-3, 10**-2, 10**-1])
plt.title(r'Relative errors for $d = 10$', y=1.03)
plt.xlabel(r'$m$')
plt.ylabel(r'$|| \Xi_{\textrm{exact}} - \Xi ||~/~|| \Xi_{\textrm{exact}} ||$')
plt.legend(loc=1, fontsize=12).get_frame().set_alpha(1)
plt.savefig('example_fpu_1_b.pdf', dpi='figure', format='pdf')
plt.show()
