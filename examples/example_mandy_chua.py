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
from mpl_toolkits.mplot3d import Axes3D

D = scipy.io.loadmat("../data/ChuaData.mat")  # load data
y = D["y"]
ya1=D["ya1"]
ya2=D["ya2"]
# plots

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams.update({'font.size': 18})
plt.rcParams.update({'figure.autolayout': True})

fig = plt.figure(dpi=300)
ax = fig.gca(projection='3d')

ax.plot(y[0,:],y[1,:],y[2,:],label=r'original system')
ax.plot(y[0,:],y[1,:],y[2,:],'r',label=r'identified system')

plt.legend(loc=1).get_frame().set_alpha(1)
#plt.grid()
plt.xticks(np.arange(-2,4,2))
ax.set_yticks([-0.25, int(0), 0.25])
ax.set_yticklabels(['-0.25','0','0.25'])
ax.set_zticks(np.arange(-2,4,2))
ax.xaxis.labelpad=15
ax.yaxis.labelpad=15
ax.zaxis.labelpad=5

ax.xaxis.set_rotate_label(False)
ax.yaxis.set_rotate_label(False)
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_zlabel(r'$x_3$')
plt.savefig('example_chua_2.pdf', dpi='figure', format='pdf', bbox_inches='tight')
plt.show()

fig = plt.figure(dpi=300)
ax = fig.gca(projection='3d')

ax.plot(ya2[0,:],ya2[1,:],ya2[2,:],label=r'original system')
ax.plot(ya1[0,:],ya1[1,:],ya1[2,:],'r',label=r'identified system')

plt.legend(loc=1).get_frame().set_alpha(1)
#plt.grid()
plt.xticks(np.arange(-2,4,2))
ax.set_yticks([-0.25, int(0), 0.25])
ax.set_yticklabels(['-0.25','0','0.25'])
ax.set_zticks(np.arange(-2,4,2))
ax.xaxis.labelpad=15
ax.yaxis.labelpad=15
ax.zaxis.labelpad=5

ax.xaxis.set_rotate_label(False)
ax.yaxis.set_rotate_label(False)
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.set_zlabel(r'$x_3$')
plt.savefig('example_chua_1.pdf', dpi='figure', format='pdf', bbox_inches='tight')
plt.show()


#
# npzfile = np.load('../data/example_mandy_fpu_1.npz') # load data
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




